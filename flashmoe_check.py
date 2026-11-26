#!/usr/bin/env python3
"""Observe the FlashMoE reduced kernels: plan replay and a reference comparison.

For each role this script
  1. runs the complete reduced program (kernels/flashmoe-program-<role>.cu);
  2. replays the recorded host call plan (proofs/flashmoe-program/<role>/plan.json)
     family by family, running each reduced family kernel
     (kernels/flashmoe-<family>-<role>.cu) and the Kuiper-extracted kernel of its
     verified implementation (kernels/flashmoe-extracted/<family>-<role>/) on the
     same arguments, and compares their outputs bitwise;
  3. compares the plan's final outputs bitwise with the complete program's;
  4. compares the program's outputs with reference/flashmoe/base.py reference_route.
Base and optimized decompose the router into different families.

These are finite observations on the chosen inputs, not proofs.
"""

import argparse
import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import types

from mirage_check import kuiper_headers, resolve, run, same_bits
from replay import ROOT, inventory, read_json, sha256, write_json
from run_kernels import compile_kernel

EXTRACTED = ROOT / "kernels/flashmoe-extracted"
TOKENS, HIDDEN, EXPERTS, CAPACITY = 256, 512, 8, 32


def make_inputs(torch, seed, device):
    """Small dyadic BF16 inputs inside input-domain.txt (|x| <= 32, |w| <= 4) with one hot expert."""
    generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"flashmoe-check|{seed}".encode()).digest()[:8], "little"))
    x = torch.randint(-4, 5, (TOKENS, HIDDEN), generator=generator).float() / 16
    w = torch.randint(-4, 5, (HIDDEN, EXPERTS), generator=generator).float() / 4096
    hot = int(torch.randint(EXPERTS, (), generator=generator))
    x[:, 0] = 1
    w[0] = 0
    w[0, hot] = 4
    ids = torch.arange(TOKENS)
    x[:, -2] = (ids % 128).float() / 128
    x[:, -1] = (ids // 128).float() / 8
    w[-2:] = 0
    return [value.to(torch.bfloat16).to(device).contiguous() for value in (x, w)]


def reference_route():
    """reference/flashmoe/base.py reference_route, with its case-adapter import stubbed out."""
    adapter = types.ModuleType("E8_mega_kernel_h100.cases.flashmoe.adapter")
    for name in ("CASE_ID", "environment_profile", "make_inputs", "profile", "validate_inputs"):
        setattr(adapter, name, None)
    for name in ("E8_mega_kernel_h100", "E8_mega_kernel_h100.cases", "E8_mega_kernel_h100.cases.flashmoe"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[adapter.__name__] = adapter
    spec = importlib.util.spec_from_file_location("flashmoe_reference", ROOT / "reference/flashmoe/base.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_route


def check_role(torch, role, cases, work, jobs, headers, inputs):
    plan = read_json(ROOT / f"proofs/flashmoe-program/{role}/plan.json")
    program_case = cases[f"flashmoe-program-{role}"]
    if plan["accepted_cuda_sha256"] != sha256(ROOT / program_case["kernel"]):
        raise ValueError("The plan does not bind the shipped program CUDA")
    print(f"[{role}] compiling program and families", flush=True)
    program, _ = compile_kernel(torch, program_case, work, jobs)
    reduced, extracted = {}, {}
    include = [headers / "include", headers / "inst/include/krml"]
    compat = headers / "inst/include/krml/krml/internal/compat.h"
    for family in sorted(plan["families"]):
        case = cases[f"flashmoe-{family}-{role}"]
        reduced[family], _ = compile_kernel(torch, case, work, jobs)
        directory = EXTRACTED / f"{family}-{role}"
        extracted[family], _ = compile_kernel(
            torch, {**case, "id": case["id"] + "-extracted"}, work, jobs,
            source=directory / "binding.cu", include_paths=[directory, *include],
            cuda_flags=["--fmad=false", "-include", str(compat)],
        )
    print(f"[{role}] running the complete program", flush=True)
    program_outputs, program_timing = run(torch, program, inputs)
    steps, state = [], {}
    for step in plan["cases"]["sample-0000"]["steps"]:
        arguments = [resolve(torch, binding, inputs, state) for binding in step["arguments"]]
        ours, reduced_timing = run(torch, reduced[step["family"]], arguments)
        theirs, extracted_timing = run(torch, extracted[step["family"]], arguments)
        equal = len(ours) == len(theirs) and all(same_bits(torch, a, b) for a, b in zip(ours, theirs))
        steps.append({"id": step["id"], "family": step["family"], "extracted_bitwise_equal": equal,
                      "reduced_cuda_ms": reduced_timing["cuda_ms"],
                      "extracted_cuda_ms": extracted_timing["cuda_ms"]})
        if not equal:
            print(f"[{role}] {step['id']}: extracted != reduced", flush=True)
        state[step["id"]] = ours
    plan_outputs = [resolve(torch, binding, inputs, state) for binding in plan["cases"]["sample-0000"]["outputs"]]
    composed = len(plan_outputs) == len(program_outputs) and all(
        same_bits(torch, a, b) for a, b in zip(plan_outputs, program_outputs))
    by_family = {}
    for step in steps:
        entry = by_family.setdefault(step["family"], {"calls": 0, "extracted_bitwise_equal": 0})
        entry["calls"] += 1
        entry["extracted_bitwise_equal"] += step["extracted_bitwise_equal"]
    print(f"[{role}] families {by_family}; composed == complete program: {composed}", flush=True)
    return {
        "role": role, "calls": len(steps), "families": by_family,
        "all_extracted_bitwise_equal": all(step["extracted_bitwise_equal"] for step in steps),
        "composed_equals_program": composed, "program_cuda_ms": program_timing["cuda_ms"],
        "reduced_family_cuda_ms_total": sum(step["reduced_cuda_ms"] for step in steps),
        "extracted_family_cuda_ms_total": sum(step["extracted_cuda_ms"] for step in steps),
    }, program_outputs


def compare_reference(torch, outputs, expected):
    demand, indices, probabilities, survival = outputs
    difference = (probabilities.cpu() - expected["probabilities"].cpu()).abs()
    return {
        "expert_demand_equal": bool(torch.equal(demand.cpu(), expected["expert_demand"].cpu())),
        "token_indices_equal": bool(torch.equal(indices.cpu(), expected["token_indices"].cpu())),
        "survival_equal": bool(torch.equal(survival.cpu(), expected["survival"].cpu())),
        "probabilities_max_abs_difference": difference.max().item(),
        "expert_demand": demand.cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("both", "base", "optimized"), default="both")
    parser.add_argument("--seed", type=int, action="append", help="Input seed; repeatable (default 0)")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent compiler processes")
    parser.add_argument("--work", type=Path, default=ROOT / ".work/flashmoe")
    args = parser.parse_args()
    seeds = args.seed or [0]
    manifest = inventory()
    cases = {case["id"]: case for case in manifest["cases"]}
    if "CUDA_HOME" not in os.environ and shutil.which("nvcc"):
        os.environ["CUDA_HOME"] = str(Path(shutil.which("nvcc")).resolve().parent.parent)
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install the documented GPU dependencies.")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    headers = kuiper_headers(args.work)
    shutil.copytree(EXTRACTED / "include", headers / "include", dirs_exist_ok=True)
    route = reference_route()
    results = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "scope": "finite observations on the chosen inputs; not proofs", "samples": [],
    }
    roles = ("base", "optimized") if args.role == "both" else (args.role,)
    for seed in seeds:
        inputs = make_inputs(torch, seed, device)
        expected = route(inputs[0], inputs[1], CAPACITY)
        sample = {"seed": seed, "roles": [], "reference": {}}
        for role in roles:
            result, outputs = check_role(torch, role, cases, args.work, args.jobs, headers, inputs)
            sample["roles"].append(result)
            sample["reference"][role] = compare_reference(torch, outputs, expected)
            print(f"[{role}] reference: {sample['reference'][role]}", flush=True)
        results["samples"].append(sample)
        write_json(args.work / "results.json", results)
    passed = all(role["all_extracted_bitwise_equal"] and role["composed_equals_program"]
                 for sample in results["samples"] for role in sample["roles"])
    results["status"] = "passed" if passed else "failed"
    results["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(args.work / "results.json", results)
    print(f"Results: {args.work / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
