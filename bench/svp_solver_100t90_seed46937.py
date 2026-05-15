#!/usr/bin/env python3
"""Reproduce the dim-100 seed-46937 SVP solver benchmark."""

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_URL = "https://www.latticechallenge.org/svp-challenge/download/generator.zip"
CHALLENGE_EXAMPLE_URL = (
    "https://www.latticechallenge.org/svp-challenge/download/challenges/"
    "svpchallengedim{dim}seed0.txt"
)
SOLVER_RE = re.compile(
    r"possible sol: ([^,]+),(?: [^,]+ = [^,]+,)* length = ([0-9.eE+-]+), time = ([0-9.eE+-]+)s"
)
CSV_FIELDS = [
    "timestamp",
    "mode",
    "dimension",
    "lattice_seed",
    "sampler_seed",
    "algo",
    "threads",
    "bkz_pre_msd",
    "bkz_pre_d4f",
    "bkz_pre_minsd",
    "returncode",
    "timed_out",
    "elapsed_sec",
    "solver_time_sec",
    "solution_norm",
    "gh",
    "challenge_bound",
    "challenge_pass",
    "approx_factor",
    "command",
    "env",
    "lattice",
    "log_file",
]


def parse_args(argv):
    default_out = ROOT / "bench" / "results" / time.strftime(
        "svp_solver_100t90_seed46937_%Y%m%d_%H%M%S", time.gmtime()
    )
    parser = argparse.ArgumentParser(
        description="Run the official dim-100 seed-46937 SVP_100T90 solver benchmark."
    )
    parser.add_argument("--mode", choices=["cpu", "cuda", "both"], default="cuda")
    parser.add_argument("--dimension", type=int, default=100)
    parser.add_argument("--lattice-seed", type=int, default=46937)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--algo", default="100t90")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--goal", type=float, default=2600.0)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--bkz-pre",
        nargs=2,
        type=int,
        metavar=("MSD", "D4F"),
        help="pass --bkz-pre MSD D4F to svp_solver before the selected reduction schedule",
    )
    parser.add_argument("--bkz-pre-minsd", type=int, default=45)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--app", type=Path, default=ROOT / "app" / "svp_solver")
    parser.add_argument("--lattice", type=Path, help="existing raw challenge lattice")
    parser.add_argument("--generator-bin", type=Path, help="existing official generator binary")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("BGJ_CUDA_DEVICES"),
        help="comma-separated runtime CUDA device ordinals for BGJ_CUDA_DEVICES, e.g. 0,1",
    )
    parser.add_argument(
        "--lll-backend",
        default=os.environ.get("BGJ_LLL_BACKEND"),
        help="initial integer LLL backend to pass as BGJ_LLL_BACKEND, e.g. fplll or ntl",
    )
    parser.add_argument(
        "--tail-lll-backend",
        default=os.environ.get("BGJ_TAIL_LLL_BACKEND"),
        help="tail_LLL regular LLL backend to pass as BGJ_TAIL_LLL_BACKEND, e.g. fplll or custom",
    )
    parser.add_argument(
        "--tail-deep-lll-backend",
        default=os.environ.get("BGJ_TAIL_DEEP_LLL_BACKEND"),
        help="tail_LLL deep pass backend to pass as BGJ_TAIL_DEEP_LLL_BACKEND, e.g. custom or skip",
    )
    parser.add_argument(
        "--cuda-overlap-cred",
        default=os.environ.get("BGJ_CUDA_OVERLAP_CRED"),
        help="pass BGJ_CUDA_OVERLAP_CRED to the CUDA solver, e.g. 0 or 1",
    )
    parser.add_argument("--validate-generator", action="store_true")
    parser.add_argument("--profile", action="store_true", help="pass --profile to svp_solver")
    parser.add_argument(
        "--require-challenge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail unless the printed vector satisfies the SVP challenge bound",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.app = args.app.resolve()
    if args.lattice is not None:
        args.lattice = args.lattice.resolve()
    if args.generator_bin is not None:
        args.generator_bin = args.generator_bin.resolve()
    return args


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BGJ-Sieve-AMX benchmark"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def matrix_integer_tokens(text):
    return [int(value) for value in re.findall(r"-?\d+", text)]


def validate_lattice_text(text, dim):
    rows = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "]":
            continue
        if not line.startswith("[") or not line.endswith("]"):
            raise ValueError(f"unexpected matrix row: {line[:80]}")
        values = line.replace("[", " ").replace("]", " ").split()
        if len(values) != dim:
            raise ValueError(f"expected {dim} columns, found {len(values)}")
        for value in values:
            int(value)
        rows += 1
    if rows != dim:
        raise ValueError(f"expected {dim} rows, found {rows}")


def bundled_dep(path):
    dep = ROOT / path
    if not dep.exists():
        raise FileNotFoundError(f"missing {dep}; run the dependency install scripts first")
    return dep


def build_generator(args):
    if args.generator_bin is not None:
        if not args.generator_bin.exists():
            raise FileNotFoundError(args.generator_bin)
        return args.generator_bin

    work = args.output_dir / "generator"
    binary = work / "generate_random"
    if binary.exists():
        return binary

    zip_path = work / "generator.zip"
    extract_dir = work / "extract"
    if not zip_path.exists():
        print(f"downloading {GENERATOR_URL}", file=sys.stderr, flush=True)
        download(GENERATOR_URL, zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    sources = list(extract_dir.rglob("generate_random.cpp"))
    if not sources:
        raise FileNotFoundError("generate_random.cpp not found in generator.zip")
    source = sources[0]
    source_dir = source.parent
    ntl = bundled_dep("dep/ntl")
    gmp = bundled_dep("dep/gmp")
    command = [
        args.cxx,
        "-O2",
        "-std=c++17",
        "-stdlib=libc++",
        "-pthread",
        f"-I{ntl / 'include'}",
        f"-I{gmp / 'include'}",
        str(source),
        f"-L{ntl / 'lib'}",
        "-lntl",
        f"-L{gmp / 'lib'}",
        "-lgmp",
        "-lm",
        f"-Wl,-rpath,{ntl / 'lib'}",
        f"-Wl,-rpath,{gmp / 'lib'}",
        "-o",
        str(binary),
    ]
    print(f"building generator: {shlex.join(command)}", file=sys.stderr, flush=True)
    subprocess.run(command, cwd=str(source_dir), check=True)
    return binary


def generate_lattice(generator, dim, seed, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(generator), "--dim", str(dim), "--seed", str(seed)]
    print(f"generating lattice: {shlex.join(command)}", file=sys.stderr, flush=True)
    output = subprocess.check_output(command, text=True)
    validate_lattice_text(output, dim)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(output.rstrip() + "\n", encoding="ascii")
    tmp.replace(target)


def validate_generator(generator, dim):
    expected_url = CHALLENGE_EXAMPLE_URL.format(dim=dim)
    expected = urllib.request.urlopen(expected_url, timeout=120).read().decode("ascii")
    generated = subprocess.check_output(
        [str(generator), "--dim", str(dim), "--seed", "0"], text=True
    )
    if matrix_integer_tokens(expected) != matrix_integer_tokens(generated):
        raise ValueError(f"generator output does not match downloadable dim-{dim} seed-0 example")
    print(f"validated generator against dim-{dim} seed-0 downloadable example", file=sys.stderr)


def prepare_lattice(args):
    target = args.output_dir / "raw" / f"L_{args.dimension}_{args.lattice_seed}"
    if args.lattice is not None:
        text = args.lattice.read_text(encoding="ascii")
        validate_lattice_text(text, args.dimension)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip() + "\n", encoding="ascii")
        return target
    if target.exists():
        return target
    generator = build_generator(args)
    if args.validate_generator:
        validate_generator(generator, args.dimension)
    generate_lattice(generator, args.dimension, args.lattice_seed, target)
    return target


def gaussian_heuristic(raw_lattice, dim):
    first_value = matrix_integer_tokens(raw_lattice.read_text(encoding="ascii"))[0]
    det_root = math.exp(math.log(first_value) / dim)
    coeff = math.exp(math.lgamma(dim / 2.0 + 1.0) / dim) / math.sqrt(math.pi)
    return coeff * det_root


def run_solver(args, mode, lattice, gh):
    env_update = {}
    command = [
        str(args.app),
        lattice.name,
        "--algo",
        args.algo,
        "--threads",
        str(args.threads),
        "--seed",
        str(args.sampler_seed),
        "--goal",
        str(args.goal),
    ]
    if args.bkz_pre is not None:
        command.extend(["--bkz-pre", str(args.bkz_pre[0]), str(args.bkz_pre[1])])
        if args.bkz_pre_minsd != 45:
            command.extend(["--bkz-pre-minsd", str(args.bkz_pre_minsd)])
    if mode == "cuda":
        command.append("--cuda")
        if args.cuda_visible_devices:
            env_update["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        if args.cuda_devices:
            env_update["BGJ_CUDA_DEVICES"] = args.cuda_devices
    if args.lll_backend:
        env_update["BGJ_LLL_BACKEND"] = args.lll_backend
    if args.tail_lll_backend:
        env_update["BGJ_TAIL_LLL_BACKEND"] = args.tail_lll_backend
    if args.tail_deep_lll_backend:
        env_update["BGJ_TAIL_DEEP_LLL_BACKEND"] = args.tail_deep_lll_backend
    if args.cuda_overlap_cred:
        env_update["BGJ_CUDA_OVERLAP_CRED"] = args.cuda_overlap_cred
    if args.profile:
        command.append("--profile")

    log_file = args.output_dir / "logs" / f"{mode}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "dimension": args.dimension,
        "lattice_seed": args.lattice_seed,
        "sampler_seed": args.sampler_seed,
        "algo": args.algo,
        "threads": args.threads,
        "bkz_pre_msd": args.bkz_pre[0] if args.bkz_pre is not None else "",
        "bkz_pre_d4f": args.bkz_pre[1] if args.bkz_pre is not None else "",
        "bkz_pre_minsd": args.bkz_pre_minsd if args.bkz_pre is not None else "",
        "gh": gh,
        "challenge_bound": 1.05 * gh,
        "command": shlex.join(command),
        "env": json.dumps(env_update, sort_keys=True),
        "lattice": str(lattice),
        "log_file": str(log_file),
    }

    if args.dry_run:
        row.update(
            {
                "returncode": "",
                "timed_out": False,
                "elapsed_sec": 0.0,
                "solver_time_sec": 0.0,
                "solution_norm": 0.0,
                "challenge_pass": "",
                "approx_factor": 0.0,
            }
        )
        return row

    env = os.environ.copy()
    env.update(env_update)
    start = time.monotonic()
    timed_out = False
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(args.output_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            returncode = proc.wait(timeout=args.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            returncode = 124
            log.write(f"\n[TIMEOUT after {args.timeout_sec}s]\n")
    elapsed = time.monotonic() - start
    output = log_file.read_text(encoding="utf-8", errors="replace")
    match = SOLVER_RE.search(output)
    norm = float(match.group(2)) if match else 0.0
    solver_time = float(match.group(3)) if match else 0.0
    approx = norm / gh if norm > 0.0 and gh > 0.0 else 0.0
    row.update(
        {
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_sec": elapsed,
            "solver_time_sec": solver_time,
            "solution_norm": norm,
            "challenge_pass": bool(norm > 0.0 and norm <= 1.05 * gh),
            "approx_factor": approx,
        }
    )
    return row


def write_results(args, rows):
    csv_path = args.output_dir / "results.csv"
    jsonl_path = args.output_dir / "results.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for row in rows:
            jsonl.write(json.dumps(row, sort_keys=True) + "\n")
    return csv_path, jsonl_path


def main(argv):
    args = parse_args(argv)
    if not args.app.exists() and not args.dry_run:
        print(f"missing solver binary: {args.app}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "red").mkdir(parents=True, exist_ok=True)
    lattice = prepare_lattice(args)
    gh = gaussian_heuristic(lattice, args.dimension)
    modes = ["cpu", "cuda"] if args.mode == "both" else [args.mode]
    rows = [run_solver(args, mode, lattice, gh) for mode in modes]
    csv_path, jsonl_path = write_results(args, rows)

    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")
    print("mode,elapsed_sec,solver_time_sec,norm,bound,approx,pass,returncode,timeout")
    failed = False
    for row in rows:
        print(
            ",".join(
                [
                    str(row["mode"]),
                    f"{float(row['elapsed_sec'] or 0.0):.3f}",
                    f"{float(row['solver_time_sec'] or 0.0):.3f}",
                    f"{float(row['solution_norm'] or 0.0):.6f}",
                    f"{float(row['challenge_bound'] or 0.0):.6f}",
                    f"{float(row['approx_factor'] or 0.0):.9f}",
                    str(row["challenge_pass"]),
                    str(row["returncode"]),
                    str(row["timed_out"]),
                ]
            )
        )
        if args.require_challenge and not args.dry_run and not row["challenge_pass"]:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
