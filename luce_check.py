#!/usr/bin/env python3
"""Observe the LUCE reduced kernels: plan replay and a base/optimized comparison.

For each role and each recorded plan sample this script
  1. runs the complete reduced program (kernels/luce-program-<role>.cu);
  2. replays the recorded host call plan (proofs/luce-program/<role>/plan.json)
     family by family, running each reduced family kernel
     (kernels/luce-<family>-<role>.cu) and the Kuiper-extracted kernel of its
     verified implementation (kernels/luce-extracted/<family>-<role>/) on the
     same arguments, and compares their outputs bitwise;
  3. compares the plan's final outputs bitwise with the complete program's.
It then records how far the base and optimized outputs agree. They need not be
equal: both only approximate the same real-valued reference.

The reduced kernels are built with nvcc's default flags (FMA contraction on, as
when RESOLVE recorded them); the extracted kernels with --fmad=false, as Kuiper
validated them. The optimized attention decoder's extracted code writes the
contracted fma explicitly.

Weights are synthetic (seeded) unless --weights gives the packed BF16 buffer
(raw 752393024-element file, or a Qwen/Qwen3.5-0.8B safetensors file packed per
reference/luce/weight_layout.json). These are finite observations, not proofs.
"""

import argparse
import datetime
import os
from pathlib import Path
import shutil
import sys

from mirage_check import kuiper_headers, resolve, run, same_bits
from replay import ROOT, inventory, read_json, sha256, write_json
from run_kernels import compile_kernel

EXTRACTED = ROOT / "kernels/luce-extracted"
COUNT = 752393024
# The prompts and decode steps of RESOLVE's four testing samples, in plan order.
SAMPLES = {
    "sample-0000": ([1234, 5678, 9012], 5),
    "sample-0001": ([0], 1),
    "sample-0002": ([65525, 24019, 205255, 0, 42020, 21425, 130170, 130170], 5),
    "sample-0003": ([122570, 141810], 1),
}


def load_weights(torch, args, device):
    layout = read_json(ROOT / "reference/luce/weight_layout.json")
    if args.weights is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        weights = torch.randn(COUNT, generator=generator, device=device) * 0.02
        for item in layout["parameters"]:
            if item["name"].endswith("linear_attn.norm.weight"):  # applied as weight, not 1 + weight
                weights[item["offset"]:item["offset"] + item["elements"]] += 1
        return weights.to(torch.bfloat16), {"kind": "synthetic", "seed": args.seed}
    if args.weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(str(args.weights), device="cpu")
        weights = torch.empty(COUNT, dtype=torch.bfloat16)
        for item in layout["parameters"]:
            name = item["name"]
            if name not in tensors:  # the multimodal checkpoint nests the text model
                name = name.replace("model.", "model.language_model.", 1)
            value = tensors[name].to(torch.bfloat16).reshape(-1)
            weights[item["offset"]:item["offset"] + item["elements"]] = value
        kind = "safetensors"
    else:
        weights = torch.from_file(str(args.weights), size=COUNT, dtype=torch.bfloat16)
        kind = "packed"
    return weights.to(device).contiguous(), {"kind": kind, "sha256": sha256(args.weights)}


def check_role(torch, role, cases, work, jobs, headers, weights, device):
    plan = read_json(ROOT / f"proofs/luce-program/{role}/plan.json")
    program_case = cases[f"luce-program-{role}"]
    if plan["accepted_cuda_sha256"] != sha256(ROOT / program_case["kernel"]):
        raise ValueError("The plan does not bind the shipped program CUDA")
    print(f"[{role}] compiling program and families", flush=True)
    program, _ = compile_kernel(torch, program_case, work, jobs)
    reduced, extracted = {}, {}
    include = [headers / "include", headers / "inst/include/krml"]
    compat = headers / "inst/include/krml/krml/internal/compat.h"
    for family in sorted(plan["families"]):
        case = cases[f"luce-{family}-{role}"]
        reduced[family], _ = compile_kernel(torch, case, work, jobs)
        directory = EXTRACTED / f"{family}-{role}"
        extracted[family], _ = compile_kernel(
            torch, {**case, "id": case["id"] + "-extracted"}, work, jobs,
            source=directory / "binding.cu", include_paths=[directory, *include],
            cuda_flags=["--fmad=false", "-include", str(compat)],
        )
    results, outputs = [], {}
    for sample, (prompt, steps_count) in SAMPLES.items():
        inputs = [weights, torch.tensor(prompt, dtype=torch.int64, device=device), steps_count]
        program_outputs, program_timing = run(torch, program, inputs)
        steps, state = [], {}
        for step in plan["cases"][sample]["steps"]:
            arguments = [resolve(torch, binding, inputs, state) for binding in step["arguments"]]
            ours, reduced_timing = run(torch, reduced[step["family"]], arguments)
            theirs, extracted_timing = run(torch, extracted[step["family"]], arguments)
            equal = len(ours) == len(theirs) and all(same_bits(torch, a, b) for a, b in zip(ours, theirs))
            steps.append({"family": step["family"], "equal": equal,
                          "reduced_ms": reduced_timing["cuda_ms"], "extracted_ms": extracted_timing["cuda_ms"]})
            if not equal:
                print(f"[{role} {sample}] {step['id']}: extracted != reduced", flush=True)
            state[step["id"]] = ours
        plan_outputs = [resolve(torch, binding, inputs, state) for binding in plan["cases"][sample]["outputs"]]
        composed = len(plan_outputs) == len(program_outputs) and all(
            same_bits(torch, a, b) for a, b in zip(plan_outputs, program_outputs))
        by_family = {}
        for step in steps:
            entry = by_family.setdefault(step["family"], {"calls": 0, "extracted_bitwise_equal": 0})
            entry["calls"] += 1
            entry["extracted_bitwise_equal"] += step["equal"]
        print(f"[{role} {sample}] {len(steps)} calls {by_family}; composed == program: {composed}", flush=True)
        results.append({
            "sample": sample, "prompt_length": len(prompt), "decode_steps": steps_count,
            "calls": len(steps), "families": by_family,
            "all_extracted_bitwise_equal": all(step["equal"] for step in steps),
            "composed_equals_program": composed, "tokens": program_outputs[0].cpu().tolist(),
            "program_cuda_ms": program_timing["cuda_ms"],
            "reduced_family_cuda_ms_total": sum(step["reduced_ms"] for step in steps),
            "extracted_family_cuda_ms_total": sum(step["extracted_ms"] for step in steps),
        })
        outputs[sample] = program_outputs
    return results, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, help="Packed BF16 weights (raw) or Qwen3.5-0.8B .safetensors")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic weight seed")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent compiler processes")
    parser.add_argument("--work", type=Path, default=ROOT / ".work/luce")
    args = parser.parse_args()
    cases = {case["id"]: case for case in inventory()["cases"]}
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
    weights, weight_source = load_weights(torch, args, device)
    results = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "scope": "finite observations on the chosen inputs; not proofs",
        "weights": weight_source, "roles": {}, "base_vs_optimized": {},
    }
    outputs = {}
    for role in ("base", "optimized"):
        results["roles"][role], outputs[role] = check_role(
            torch, role, cases, args.work, args.jobs, headers, weights, device)
        write_json(args.work / "results.json", results)
    for sample in SAMPLES:
        (tb, hb), (to, ho) = outputs["base"][sample], outputs["optimized"][sample]
        difference = (hb.float() - ho.float()).abs()
        results["base_vs_optimized"][sample] = {
            "tokens_equal": bool(torch.equal(tb, to)), "hidden_bitwise_equal": same_bits(torch, hb, ho),
            "hidden_max_abs_difference": difference.max().item(),
            "hidden_max_abs_value": hb.float().abs().max().item(),
        }
    print(f"base vs optimized: {results['base_vs_optimized']}", flush=True)
    passed = all(entry["all_extracted_bitwise_equal"] and entry["composed_equals_program"]
                 for role in results["roles"].values() for entry in role)
    results["status"] = "passed" if passed else "failed"
    results["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(args.work / "results.json", results)
    print(f"Results: {args.work / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
