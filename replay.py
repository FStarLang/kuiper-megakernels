#!/usr/bin/env python3
"""Check the frozen artifacts, without RESOLVE, DTester, or proof discovery."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def inventory():
    manifest = read_json(ROOT / "manifest.json")
    actual = {str(p.relative_to(ROOT)) for name in ("proofs", "support", "kernels")
              for p in (ROOT / name).rglob("*") if p.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError(f"Artifact inventory mismatch: {sorted(actual ^ set(manifest['files']))}")
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Missing or changed artifact: {relative}")
    identifiers = [case["id"] for case in manifest["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate case IDs.")
    cases_by_id = {case["id"]: case for case in manifest["cases"]}
    for case in manifest["cases"]:
        roots = case["roots"]
        if (
            not roots or len(roots) != len({Path(root).name for root in roots})
            or not any(Path(root).parent == Path(case["directory"]) for root in roots)
            or any(
                root not in manifest["files"] or not Path(root).is_relative_to("proofs")
                or Path(root).suffix not in (".fst", ".fsti")
                for root in roots
            )
        ):
            raise ValueError(f"Unindexed or misplaced proof root: {case['id']}")
        if case["kind"] == "review":
            implementations = case.get("implementation_cases", [])
            if not implementations or any(
                identifier not in cases_by_id
                or cases_by_id[identifier]["kind"] not in ("program", "family")
                or cases_by_id[identifier]["profile"] != case["profile"]
                or not set(cases_by_id[identifier]["roots"]).issubset(roots)
                for identifier in implementations
            ):
                raise ValueError(f"Review omits implementation checks: {case['id']}")
    return manifest


def source_files(directory):
    return sorted(p for p in directory.rglob("*") if p.suffix in (".fst", ".fsti"))


def input_fingerprint(manifest):
    return hashlib.sha256(
        (json.dumps(manifest, sort_keys=True) + (ROOT / "toolchain.json").read_text()).encode()
    ).hexdigest()


def support_fingerprint(manifest):
    return input_fingerprint({
        "profiles": manifest["profiles"],
        "files": {p: h for p, h in manifest["files"].items() if p.startswith("support/")},
    })


def toolchain_name(manifest, profile):
    return manifest.get("toolchains", {}).get(profile, "default")


def toolchain_root(work, name):
    return work if name == "default" else work / "toolchains" / name


def prepare_toolchain(root, pin):
    archive = root / "toolchain.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print(f"Downloading pinned Kuiper toolchain {pin['url']}...", flush=True)
        with urllib.request.urlopen(pin["url"], timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(archive)
    if sha256(archive) != pin["sha256"]:
        raise ValueError(f"Toolchain checksum mismatch: {archive}")
    destination = root / "toolchain"
    if destination.exists():
        raise ValueError("Incomplete preparation; use a new --work directory.")
    with tempfile.TemporaryDirectory(dir=root, prefix="unpack-") as temporary:
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        Path(temporary).rename(destination)
    package = destination / "kuiper"
    version = subprocess.check_output(
        [str(package / "inst/bin/fstar.exe"), "--version"], text=True
    )
    if f"commit={pin['fstar_commit']}" not in version:
        raise ValueError("Unexpected F* compiler commit in downloaded package.")
    stdlib = root / "stdlib"
    shutil.copytree(package / "inst/lib/fstar", stdlib)
    if pin.get("foundation"):
        for source in source_files(ROOT / "support/foundation"):
            matches = list(stdlib.rglob(source.name))
            if len(matches) > 1:
                raise ValueError(f"Ambiguous foundational source: {source.name}")
            target = matches[0] if matches else stdlib / "ulib" / source.name
            shutil.copyfile(source, target)
            for checked in stdlib.rglob(source.name + ".checked"):
                checked.unlink()
    return version


def prepare(work, manifest):
    pins = read_json(ROOT / "toolchain.json")
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise ValueError("The pinned binary toolchain requires Linux x86_64.")
    work.mkdir(parents=True, exist_ok=True)
    state = work / "prepared.json"
    fingerprint = support_fingerprint(manifest)
    if state.exists():
        if read_json(state)["support_sha256"] != fingerprint:
            raise ValueError("Inputs changed; use a new --work directory for preparation.")
        return
    names = sorted({toolchain_name(manifest, profile) for profile in manifest["profiles"]})
    versions = {}
    for name in names:
        root = toolchain_root(work, name)
        root.mkdir(parents=True, exist_ok=True)
        versions[name] = prepare_toolchain(root, pins[name])
    for profile, overlays in manifest["profiles"].items():
        package = toolchain_root(work, toolchain_name(manifest, profile)) / "toolchain/kuiper"
        library = work / "support" / profile / "sources"
        library.mkdir(parents=True)
        for source in source_files(package / "src/lib"):
            target = library / source.name
            if target.exists():
                raise ValueError(f"Duplicate library module: {source.name}")
            shutil.copyfile(source, target)
        for overlay in overlays:
            for source in source_files(ROOT / "support" / overlay):
                shutil.copyfile(source, library / source.name)
    write_json(state, {"support_sha256": fingerprint, "fstar_version": versions,
                       "toolchain": {name: pins[name] for name in names}})


def command(work, profile, cache, source_directories, toolchain="default"):
    root = toolchain_root(work, toolchain)
    package = root / "toolchain/kuiper"
    z3 = read_json(ROOT / "toolchain.json")[toolchain]["z3_version"]
    stdlib = root / "stdlib"
    includes = [stdlib, stdlib / "pulse", stdlib / "fstarc"]
    includes += [work / "support" / profile / "sources"]
    includes += [*source_directories, cache]
    result = [
        str(package / "inst/bin/fstar.exe"),
        "--no_default_includes", "--cache_dir", str(cache), "--odir", str(cache),
        "--force", "--cache_off",
        "--admit_smt_queries", "false",
        "--smt", str(package / f"inst/bin/z3-{z3}"), "--z3version", z3,
        "--z3rlimit", "0", "--z3cliopt", "timeout=0",
        "--warn_error", "-274-291-249-321-288-271", "--warn_error", "@242@250",
        "--query_stats",
    ]
    for extension in ("kuiper", "__unrefine", "no_krml_private", "pulse:extra_simplify",
                      "context_pruning_no_ambients", "freshen"):
        result += ["--ext", extension]
    for include in includes:
        result += ["--include", str(include)]
    return result


def check_module(config_path, source):
    config = read_json(config_path)
    source = Path(source)
    output = Path(config["output"])
    cmd = config["command"]
    scratch = output / "scratch" / source.name
    scratch.mkdir(parents=True, exist_ok=True)
    for option in ("--cache_dir", "--odir"):
        cmd[cmd.index(option) + 1] = str(scratch)
    started = time.monotonic()
    with (output / (source.name + ".log")).open("w") as log:
        process = subprocess.run(cmd + [str(source)],
                                 stdout=log, stderr=subprocess.STDOUT)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "module": source.name, "source_sha256": sha256(source),
        "wall_seconds": time.monotonic() - started,
        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss, "returncode": process.returncode,
    }
    write_json(output / (source.name + ".json"), record)
    print(f"{'PASS' if process.returncode == 0 else 'FAIL'} {source.name} "
          f"{record['wall_seconds']:.3f}s", flush=True)
    if process.returncode:
        print((output / (source.name + ".log")).read_text(errors="replace")[-8000:],
              file=sys.stderr)
    return process.returncode


def check_set(work, profile, sources, output, jobs, toolchain="default"):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "checked"
    cache.mkdir(exist_ok=True)
    source_directories = list(dict.fromkeys(source.parent for source in sources))
    cmd = command(work, profile, cache, source_directories, toolchain)
    config = output / "command.json"
    write_json(config, {"command": cmd, "output": str(output)})
    started = time.monotonic()
    def run(source):
        return subprocess.call([sys.executable, str(Path(__file__).resolve()),
                                "_check", str(config), str(source)])
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        statuses = list(pool.map(run, sources))
    modules = [read_json(output / (p.name + ".json")) for p in sources]
    result = {
        "status": "passed" if all(code == 0 for code in statuses) else "failed",
        "wall_seconds": time.monotonic() - started, "jobs": jobs,
        "requested_modules": len(sources), "modules": modules,
    }
    write_json(output / "result.json", result)
    return result


def setup(work, manifest, profiles, jobs):
    prepare(work, manifest)
    for profile in sorted(profiles):
        output = work / "support" / profile
        if (output / "result.json").exists():
            if read_json(output / "result.json")["status"] == "passed":
                continue
        names = {p.name for overlay in manifest["profiles"][profile]
                 for p in source_files(ROOT / "support" / overlay)}
        sources = [output / "sources" / name for name in sorted(names)]
        toolchain = toolchain_name(manifest, profile)
        if read_json(ROOT / "toolchain.json")[toolchain].get("foundation"):
            stdlib = toolchain_root(work, toolchain) / "stdlib"
            for source in source_files(ROOT / "support/foundation"):
                sources += list(stdlib.rglob(source.name))
        print(f"Checking support library: {profile}", flush=True)
        result = check_set(work, profile, sources, output, jobs, toolchain)
        if result["status"] != "passed":
            raise RuntimeError(f"Support verification failed: {profile}")


def verify(args, manifest):
    cases = [
        case for case in manifest["cases"]
        if (args.kind == "all" or case["kind"] == args.kind)
        and (not args.case or any(fnmatch.fnmatchcase(case["id"], p) for p in args.case))
    ]
    if not cases:
        raise ValueError("No matching cases.")
    setup(args.work, manifest, {case["profile"] for case in cases}, args.jobs)
    name = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run = args.work / "runs" / name
    run.mkdir(parents=True)
    results = {
        "status": "running",
        "requested_cases": [case["id"] for case in cases],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "jobs": args.jobs,
        "input_sha256": input_fingerprint(manifest),
        "toolchain": read_json(args.work / "prepared.json"),
        "cache_policy": "force/cache_off for every module; setup support checks recorded separately",
        "cases": [],
    }
    started = time.monotonic()
    write_json(run / "results.json", results)
    for case in cases:
        print(f"Checking {case['id']}", flush=True)
        outcome = check_set(args.work, case["profile"], [ROOT / p for p in case["roots"]],
                            run / case["id"], args.jobs, toolchain_name(manifest, case["profile"]))
        results["cases"].append({"id": case["id"], "kind": case["kind"], **outcome})
        results["wall_seconds"] = time.monotonic() - started
        write_json(run / "results.json", results)
    passed = all(case["status"] == "passed" for case in results["cases"])
    results["status"] = "passed" if passed else "failed"
    results["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(run / "results.json", results)
    print(f"Results: {run / 'results.json'}")
    return 0 if passed else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_check":
        return check_module(Path(sys.argv[2]), sys.argv[3])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "list", "integrity", "verify"))
    parser.add_argument("--work", type=Path, default=ROOT / ".work/proofs")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--kind", choices=("all", "program", "shared", "family", "tightness", "review"), default="all")
    parser.add_argument("--case", action="append", help="Case ID or shell-style pattern; repeatable")
    args = parser.parse_args()
    args.work = args.work.resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    manifest = inventory()
    if args.action == "list":
        for case in manifest["cases"]:
            print(f"{case['id']:55s} {case['kind']:10s} {len(case['roots'])} modules")
    elif args.action == "setup":
        setup(args.work, manifest, manifest["profiles"], args.jobs)
    elif args.action == "verify":
        return verify(args, manifest)
    else:
        print(f"Verified {len(manifest['files'])} artifact hashes.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
