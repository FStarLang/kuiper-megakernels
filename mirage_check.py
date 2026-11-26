#!/usr/bin/env python3
"""Observe the Mirage reduced kernels: plan replay and an optional reference check.

For each role this script
  1. runs the complete reduced program (kernels/mirage-program-<role>.cu);
  2. replays the recorded host call plan (proofs/mirage-program/<role>/plan.json)
     family by family, running each reduced family kernel
     (kernels/mirage-<family>-<role>.cu) and the Kuiper-extracted kernel of its
     verified implementation (kernels/mirage-extracted/<family>-<role>/) on the
     same arguments, and compares their outputs bitwise;
  3. compares the plan's final outputs bitwise with the complete program's.
With --checkpoint and --reference it also runs the pinned PyTorch reference
(reference/mirage/base.py) and reports how close each role's outputs are.

These are finite observations on the chosen inputs, not proofs.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request

from replay import ROOT, inventory, read_json, sha256, write_json
from run_kernels import compile_kernel, invoke

FAMILIES = ("embedding", "decoder", "projection", "argmax")
EXTRACTED = ROOT / "kernels/mirage-extracted"


def kuiper_headers(work):
    """C headers of the pinned public Kuiper release used for the Mirage proofs."""
    pin = read_json(ROOT / "toolchain.json")["kuiper-nightly-2026-11-18"]
    root = work / "kuiper-headers"
    package = root / "kuiper"
    if not (package / "include/kuiper.h").is_file():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "kuiper.tar.gz"
        if not archive.exists() or sha256(archive) != pin["sha256"]:
            print(f"Downloading pinned Kuiper release {pin['url']}...", flush=True)
            temporary = archive.with_suffix(".part")
            with urllib.request.urlopen(pin["url"], timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(archive)
        if sha256(archive) != pin["sha256"]:
            raise ValueError("Kuiper release checksum mismatch")
        with tarfile.open(archive) as source:
            members = [m for m in source.getmembers()
                       if m.name.startswith(("kuiper/include/", "kuiper/inst/include/"))]
            source.extractall(root, members=members, filter="data")
    # RESOLVE's install overrides some release headers; overlay them so quoted includes
    # inside the release headers resolve to the same files the proofs replayed against.
    shutil.copytree(EXTRACTED / "include", package / "include", dirs_exist_ok=True)
    return package


def same_bits(torch, left, right):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left == right
    raw = lambda value: value.contiguous().reshape(-1).view(torch.uint8)
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(raw(left), raw(right))


def resolve(torch, binding, inputs, state):
    if "input" in binding:
        value = inputs[binding["input"]]
    elif "step" in binding:
        value = state[binding["step"]][binding["output"]]
    else:
        return binding["literal"]
    if "view" in binding:
        view = binding["view"]
        count = 1
        for size in view["shape"]:
            count *= size
        value = value.reshape(-1).narrow(0, view["offset"], count).reshape(view["shape"])
    return value


def run(torch, module, arguments):
    # Family state is explicit: each call gets private copies of its tensor arguments.
    copies = [value.clone() if isinstance(value, torch.Tensor) else value for value in arguments]
    outputs, timing = invoke(torch, module, copies)
    return list(outputs), timing


def load_weights(torch, args, device):
    count = 596049920
    if args.checkpoint is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        weights = torch.randn(count, generator=generator, device=device) * 0.02
        # Norm weights near one keep activations in range through 28 layers.
        sys.path.insert(0, str(ROOT / "reference/mirage"))
        from model_abi import WEIGHT_LAYOUT

        for item in WEIGHT_LAYOUT:
            if item["name"].endswith("norm.weight"):
                weights[item["offset"]:item["offset"] + item["elements"]] += 1
        return weights.to(torch.bfloat16), {"kind": "synthetic", "seed": args.seed}
    from safetensors.torch import load_file

    sys.path.insert(0, str(ROOT / "reference/mirage"))
    from model_abi import CHECKPOINT_SHA256, MODEL_ID, MODEL_REVISION, pack_weights

    digest = sha256(args.checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint is not {MODEL_ID}@{MODEL_REVISION}: {digest}")
    packed = pack_weights(load_file(str(args.checkpoint), device="cpu")).to(device).contiguous()
    return packed, {"kind": "checkpoint", "model": MODEL_ID, "revision": MODEL_REVISION,
                    "sha256": digest}


def check_role(torch, role, cases, work, jobs, headers, inputs):
    plan = read_json(ROOT / f"proofs/mirage-program/{role}/plan.json")
    program_case = cases[f"mirage-program-{role}"]
    if plan["accepted_cuda_sha256"] != sha256(ROOT / program_case["kernel"]):
        raise ValueError("The plan does not bind the shipped program CUDA")
    print(f"[{role}] compiling program and families", flush=True)
    program, _ = compile_kernel(torch, program_case, work, jobs)
    reduced, extracted = {}, {}
    include = [headers / "include", headers / "inst/include/krml"]
    compat = headers / "inst/include/krml/krml/internal/compat.h"
    for family in FAMILIES:
        case = cases[f"mirage-{family}-{role}"]
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
        print(f"[{role}] {step['id']:<22} extracted == reduced: {equal}", flush=True)
        state[step["id"]] = ours
    plan_outputs = [resolve(torch, binding, inputs, state) for binding in plan["cases"]["sample-0000"]["outputs"]]
    composed = len(plan_outputs) == len(program_outputs) and all(
        same_bits(torch, a, b) for a, b in zip(plan_outputs, program_outputs))
    print(f"[{role}] composed families == complete program: {composed}", flush=True)
    return {
        "role": role, "steps": steps,
        "all_extracted_bitwise_equal": all(step["extracted_bitwise_equal"] for step in steps),
        "composed_equals_program": composed, "program_cuda_ms": program_timing["cuda_ms"],
        "tokens": program_outputs[0].cpu().tolist(),
    }, program_outputs


def compare_reference(torch, outputs, expected):
    tokens, hidden = outputs
    reference_tokens, reference_hidden = expected
    difference = (hidden.float() - reference_hidden.float()).abs()
    return {
        "tokens_equal": bool(torch.equal(tokens, reference_tokens)),
        "reference_tokens": reference_tokens.cpu().tolist(),
        "hidden_max_abs_difference": difference.max().item(),
        "hidden_mean_abs_difference": difference.mean().item(),
        "hidden_reference_max_abs": reference_hidden.float().abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("both", "base", "optimized"), default="both")
    parser.add_argument("--prompt", default="0,1,2,3,4,5,6", help="Seven comma-separated token IDs")
    parser.add_argument("--checkpoint", type=Path, help="Qwen3-0.6B model.safetensors at the pinned revision")
    parser.add_argument("--reference", action="store_true",
                        help="Also run reference/mirage/base.py (needs transformers==4.57.1 and --checkpoint)")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the synthetic weights")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent compiler processes")
    parser.add_argument("--work", type=Path, default=ROOT / ".work/mirage")
    args = parser.parse_args()
    if args.reference and args.checkpoint is None:
        parser.error("--reference compares against real weights; pass --checkpoint")
    prompt = [int(value) for value in args.prompt.split(",")]
    if len(prompt) != 7 or not all(0 <= value < 151936 for value in prompt):
        parser.error("--prompt needs seven token IDs below 151936")
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
    weights, weight_identity = load_weights(torch, args, device)
    inputs = [weights, torch.tensor(prompt, dtype=torch.int64, device=device), 1]
    results = {
        "status": "running", "prompt": prompt, "weights": weight_identity,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "scope": "finite observations on one input; not proofs", "roles": [],
    }
    roles = ("base", "optimized") if args.role == "both" else (args.role,)
    outputs = {}
    for role in roles:
        result, outputs[role] = check_role(torch, role, cases, args.work, args.jobs, headers, inputs)
        results["roles"].append(result)
        write_json(args.work / "results.json", results)
    if args.reference:
        if device.index != 0:
            parser.error("The reference requires logical cuda:0; use CUDA_VISIBLE_DEVICES.")
        sys.path.insert(0, str(ROOT / "reference"))
        from mirage.base import Model

        started = time.monotonic()
        expected = Model()(weights, inputs[1], 1)
        results["reference"] = {"seconds": time.monotonic() - started, "roles": {
            role: compare_reference(torch, outputs[role], expected) for role in roles}}
        print(json.dumps(results["reference"], indent=2))
    passed = all(role["all_extracted_bitwise_equal"] and role["composed_equals_program"]
                 for role in results["roles"])
    results["status"] = "passed" if passed else "failed"
    results["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(args.work / "results.json", results)
    print(f"Results: {args.work / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
