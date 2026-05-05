#!/usr/bin/env python3
"""Bounded seed-42 BGJ/CUDA benchmark harness for A100 tuning."""

import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SVPCHALLENGE_DIR = ROOT.parent / "G6K-GPU-Tensor" / "svpchallenge"
SVPCHALLENGE_URL = "https://www.latticechallenge.org/svp-challenge/generator.php"

BEGIN_RE = re.compile(r"begin bgj1 sieve, sieving dimension = (\d+), pool size = (\d+)")
SOL_RE = re.compile(r"solution collect done, found (\d+) solutions in (\d+) buckets")
TIMING_RE = re.compile(
    r"^bucket time = ([0-9.]+)s, search time = ([0-9.]+)s, "
    r"sort time = ([0-9.]+)s, insert time = ([0-9.]+)s"
)
CUDA_RE = re.compile(
    r"cuda search0: single=(\d+) buckets/(\d+) dp in ([0-9.]+)s, "
    r"batch=(\d+) calls/(\d+) buckets/(\d+) dp in ([0-9.]+)s, "
    r"cred=([0-9.]+)s, fallback=(\d+) buckets/(\d+) dp in ([0-9.]+)s"
)
QUALITY_RE = re.compile(
    r"^length = ([0-9.eE+-]+)\(([0-9.eE+-]+)\)"
    r"(?:, gh = ([0-9.eE+-]+), approx = ([0-9.eE+-]+))?, vec ="
)

CSV_FIELDS = [
    "kind",
    "timestamp",
    "commit",
    "seed",
    "dim",
    "mode",
    "threads",
    "returncode",
    "timed_out",
    "elapsed_sec",
    "timeout_sec",
    "command",
    "env",
    "lattice",
    "log_file",
    "sieve_count",
    "max_sieve_dim",
    "final_pool_size",
    "solution_count",
    "solution_buckets",
    "bucket_time_sec",
    "search_time_sec",
    "sort_time_sec",
    "insert_time_sec",
    "cuda_single_buckets",
    "cuda_single_dp",
    "cuda_single_sec",
    "cuda_batch_calls",
    "cuda_batch_buckets",
    "cuda_batch_dp",
    "cuda_batch_sec",
    "cuda_cred_sec",
    "cuda_fallback_buckets",
    "cuda_fallback_dp",
    "cuda_fallback_sec",
    "cuda_gdot_per_sec",
    "quality_samples",
    "quality_euclidean_norm",
    "quality_lift_norm",
    "quality_gh",
    "quality_challenge_bound",
    "quality_challenge_pass",
    "quality_approx_factor",
    "quality_final_euclidean_norm",
    "quality_final_approx_factor",
]

PRESET_DIMS = {
    "quick": [45, 50],
    "standard": [50, 60, 70],
    "extended": [60, 70, 80, 90],
    "tune": [100],
    "ladder": [50, 60, 70, 80, 96, 100, 112],
}

PRESET_MODES = {
    "quick": ["cuda-single", "cuda-batch4", "cuda-batch8", "cuda-default"],
    "standard": ["cuda-single", "cuda-batch4", "cuda-batch8", "cuda-tensor-off", "cuda-default"],
    "extended": ["cuda-single", "cuda-batch2", "cuda-batch4", "cuda-batch8", "cuda-batch16", "cuda-tensor-off", "cuda-default"],
    "tune": ["cuda-single", "cuda-batch2", "cuda-batch4", "cuda-batch8", "cuda-batch16", "cuda-tensor-off", "cuda-default"],
    "ladder": ["cuda-batch2", "cuda-batch4", "cuda-batch8", "cuda-batch16", "cuda-tensor-off", "cuda-default"],
}

PRESET_TIMEOUTS = {
    "quick": (180, 900),
    "standard": (600, 1800),
    "extended": (1800, 7200),
    "tune": (3600, 14400),
    "ladder": (3600, 28800),
}


def parse_csv_ints(value):
    return [int(part) for part in value.split(",") if part.strip()]


def parse_csv_strings(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def repo_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def write_lattice(path, dim, seed):
    rng = random.Random((seed << 32) ^ (dim * 0x9E3779B1))
    diag_boost = max(360, dim * 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("[")
        for i in range(dim):
            row = []
            for j in range(dim):
                value = rng.randint(-100, 100)
                if i == j:
                    value += diag_boost
                row.append(str(value))
            handle.write(("[" if i == 0 else "\n[") + " ".join(row) + "]")
        handle.write("\n]\n")


def template_lattice_path(template, dim, seed):
    if not template:
        return None
    rendered = template.format(dim=dim, seed=seed)
    path = Path(rendered)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def svpchallenge_lattice_path(args, dim):
    return (
        args.svpchallenge_dir
        / f"svpchallenge-dim-{dim:03d}-seed-{args.svpchallenge_seed:02d}.txt"
    ).resolve()


def validate_svpchallenge_text(text, dim):
    rows = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "]":
            continue
        if not line.startswith("[") or not line.endswith("]"):
            raise ValueError("unexpected SVP challenge matrix row format")
        values = line.replace("[", " ").replace("]", " ").split()
        if len(values) != dim:
            raise ValueError(f"expected {dim} columns, found {len(values)}")
        for value in values:
            int(value)
        rows += 1
    if rows != dim:
        raise ValueError(f"expected {dim} rows, found {rows}")


def matrix_integer_tokens(text):
    return [int(value) for value in re.findall(r"-?\d+", text)]


def fetch_svpchallenge_lattice(args, dim):
    path = svpchallenge_lattice_path(args, dim)
    if path.exists():
        return path
    if args.dry_run:
        return path
    if not args.fetch_svpchallenge:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = urllib.parse.urlencode(
        {
            "dimension": str(dim),
            "seed": str(args.svpchallenge_seed),
            "sent": "True",
        }
    ).encode("ascii")
    request = urllib.request.Request(
        SVPCHALLENGE_URL,
        data=payload,
        headers={"User-Agent": "BGJ-Sieve-AMX benchmark harness"},
    )
    print(
        f"downloading SVP challenge dim={dim} seed={args.svpchallenge_seed}: {SVPCHALLENGE_URL}",
        file=sys.stderr,
        flush=True,
    )
    with urllib.request.urlopen(request, timeout=args.fetch_timeout_sec) as response:
        text = response.read().decode("ascii")
    validate_svpchallenge_text(text, dim)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text.rstrip() + "\n", encoding="ascii")
    tmp_path.replace(path)
    return path.resolve()


def validate_svpchallenge_generator(args, dim):
    if args.svpchallenge_seed != 0:
        raise ValueError("downloadable SVP challenge examples are seed-0 only")
    download_url = (
        "https://www.latticechallenge.org/svp-challenge/download/challenges/"
        f"svpchallengedim{dim}seed0.txt"
    )
    payload = urllib.parse.urlencode(
        {
            "dimension": str(dim),
            "seed": "0",
            "sent": "True",
        }
    ).encode("ascii")
    request_headers = {"User-Agent": "BGJ-Sieve-AMX benchmark harness"}
    with urllib.request.urlopen(
        urllib.request.Request(download_url, headers=request_headers),
        timeout=args.fetch_timeout_sec,
    ) as response:
        downloaded = response.read().decode("ascii")
    with urllib.request.urlopen(
        urllib.request.Request(SVPCHALLENGE_URL, data=payload, headers=request_headers),
        timeout=args.fetch_timeout_sec,
    ) as response:
        generated = response.read().decode("ascii")
    downloaded_values = matrix_integer_tokens(downloaded)
    generated_values = matrix_integer_tokens(generated)
    expected_values = dim * dim
    if len(downloaded_values) != expected_values:
        raise ValueError(f"downloaded dim={dim} example has {len(downloaded_values)} integers")
    if generated_values != downloaded_values:
        raise ValueError(f"SVP challenge generator did not match downloadable dim={dim} seed-0 example")
    print(
        f"validated SVP challenge generator against downloadable dim={dim} seed-0 example",
        file=sys.stderr,
        flush=True,
    )


def default_existing_lattice(args, dim):
    lattice = fetch_svpchallenge_lattice(args, dim)
    if lattice is not None and (lattice.exists() or args.dry_run):
        return lattice

    if args.allow_tmp_lattice:
        path = ROOT / "tmp" / f"L_{dim}"
        if path.exists():
            return path.resolve()
    return None


def should_preprocess_lattice(path):
    name = path.name
    if name.startswith("svpchallenge-dim-"):
        return True
    return False


def preprocess_lattice(args, source, outdir, dim):
    if not args.preprocess_svp or not should_preprocess_lattice(source):
        return source
    target = outdir / "lattices" / f"{source.stem}-lll.txt"
    if target.exists() or args.dry_run:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.preprocess_app), str(source), str(target)]
    print(f"preprocessing dim={dim}: {shlex.join(command)}", file=sys.stderr, flush=True)
    subprocess.run(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.preprocess_timeout_sec,
        check=True,
    )
    return target


def resolve_lattice(args, outdir, dim):
    lattice = template_lattice_path(args.lattice_template, dim, args.seed)
    if lattice is not None:
        if not lattice.exists():
            raise FileNotFoundError(f"lattice template resolved to missing file: {lattice}")
        return preprocess_lattice(args, lattice, outdir, dim)

    lattice = default_existing_lattice(args, dim)
    if lattice is not None:
        return preprocess_lattice(args, lattice, outdir, dim)

    if not args.synthetic_fallback:
        raise FileNotFoundError(
            f"missing SVP challenge dim={dim} seed={args.svpchallenge_seed}; "
            "enable --synthetic-fallback or --allow-tmp-lattice to use non-challenge inputs"
        )

    lattice = outdir / "lattices" / f"R_{dim}_seed{args.seed}.txt"
    if not lattice.exists() or not args.reuse_lattices:
        write_lattice(lattice, dim, args.seed)
    return lattice


def mode_command(app, lattice, mode, threads, log_level, seed):
    env = {}
    algo = "cuda"

    if mode == "cpu":
        algo = "bgj1"
    elif not mode.startswith("cuda"):
        raise ValueError(f"unknown mode: {mode}")

    if "single" in mode:
        env["BGJ_CUDA_BATCH"] = "0"

    batch_match = re.search(r"batch(\d+)", mode)
    if batch_match:
        env["BGJ_CUDA_BATCH"] = "1"
        env["BGJ_CUDA_BATCH_SIZE"] = batch_match.group(1)
        env["BGJ_CUDA_BATCH_MIN_DOTS"] = "1"

    target_match = re.search(r"target(\d+)", mode)
    if target_match:
        env["BGJ1_EPI8_BUCKET_TARGET_SIZE"] = target_match.group(1)

    if "tensor-off" in mode:
        env["BGJ_CUDA_TENSOR"] = "0"
    if "no-reorder" in mode:
        env["BGJ_CUDA_TENSOR_REORDER"] = "0"
    if "wide-off" in mode:
        env["BGJ_CUDA_TENSOR_NP_WIDE"] = "0"
    if "np-multi" in mode:
        env["BGJ_CUDA_TENSOR_NP_MULTI"] = "1"
    if "shared-a" in mode:
        env["BGJ_CUDA_TENSOR_NP_WIDE"] = "0"
        env["BGJ_CUDA_TENSOR_NP_SHARED_A"] = "1"

    np_min_match = re.search(r"np-min(\d+)", mode)
    if np_min_match:
        env["BGJ_CUDA_TENSOR_NP_MIN_TILES"] = np_min_match.group(1)

    same_min_match = re.search(r"same-min(\d+)", mode)
    if same_min_match:
        env["BGJ_CUDA_TENSOR_SAME_MIN_TILES"] = same_min_match.group(1)

    maxres_match = re.search(r"maxres(\d+)", mode)
    if maxres_match:
        env["BGJ_CUDA_MAX_RESULTS"] = maxres_match.group(1)

    command = [str(app), str(lattice), algo, str(threads), str(log_level), str(seed)]
    return command, env


def empty_app_stats():
    return {
        "sieve_count": 0,
        "max_sieve_dim": 0,
        "final_pool_size": 0,
        "solution_count": 0,
        "solution_buckets": 0,
        "bucket_time_sec": 0.0,
        "search_time_sec": 0.0,
        "sort_time_sec": 0.0,
        "insert_time_sec": 0.0,
        "cuda_single_buckets": 0,
        "cuda_single_dp": 0,
        "cuda_single_sec": 0.0,
        "cuda_batch_calls": 0,
        "cuda_batch_buckets": 0,
        "cuda_batch_dp": 0,
        "cuda_batch_sec": 0.0,
        "cuda_cred_sec": 0.0,
        "cuda_fallback_buckets": 0,
        "cuda_fallback_dp": 0,
        "cuda_fallback_sec": 0.0,
        "cuda_gdot_per_sec": 0.0,
        "quality_samples": 0,
        "quality_euclidean_norm": 0.0,
        "quality_lift_norm": 0.0,
        "quality_gh": 0.0,
        "quality_challenge_bound": 0.0,
        "quality_challenge_pass": "",
        "quality_approx_factor": 0.0,
        "quality_final_euclidean_norm": 0.0,
        "quality_final_approx_factor": 0.0,
    }


def parse_app_output(output):
    stats = empty_app_stats()
    current_cuda = None

    def flush_cuda():
        nonlocal current_cuda
        if current_cuda is None:
            return
        (
            single_buckets,
            single_dp,
            single_sec,
            batch_calls,
            batch_buckets,
            batch_dp,
            batch_sec,
            cred_sec,
            fallback_buckets,
            fallback_dp,
            fallback_sec,
        ) = current_cuda
        stats["cuda_single_buckets"] += single_buckets
        stats["cuda_single_dp"] += single_dp
        stats["cuda_single_sec"] += single_sec
        stats["cuda_batch_calls"] += batch_calls
        stats["cuda_batch_buckets"] += batch_buckets
        stats["cuda_batch_dp"] += batch_dp
        stats["cuda_batch_sec"] += batch_sec
        stats["cuda_cred_sec"] += cred_sec
        stats["cuda_fallback_buckets"] += fallback_buckets
        stats["cuda_fallback_dp"] += fallback_dp
        stats["cuda_fallback_sec"] += fallback_sec
        current_cuda = None

    for line in output.splitlines():
        begin = BEGIN_RE.search(line)
        if begin:
            flush_cuda()
            stats["sieve_count"] += 1
            sieve_dim = int(begin.group(1))
            pool_size = int(begin.group(2))
            stats["max_sieve_dim"] = max(stats["max_sieve_dim"], sieve_dim)
            stats["final_pool_size"] = pool_size
            continue

        sol = SOL_RE.search(line)
        if sol:
            stats["solution_count"] += int(sol.group(1))
            stats["solution_buckets"] += int(sol.group(2))
            continue

        timing = TIMING_RE.search(line)
        if timing:
            stats["bucket_time_sec"] += float(timing.group(1))
            stats["search_time_sec"] += float(timing.group(2))
            stats["sort_time_sec"] += float(timing.group(3))
            stats["insert_time_sec"] += float(timing.group(4))
            continue

        cuda = CUDA_RE.search(line)
        if cuda:
            current_cuda = (
                int(cuda.group(1)),
                int(cuda.group(2)),
                float(cuda.group(3)),
                int(cuda.group(4)),
                int(cuda.group(5)),
                int(cuda.group(6)),
                float(cuda.group(7)),
                float(cuda.group(8)),
                int(cuda.group(9)),
                int(cuda.group(10)),
                float(cuda.group(11)),
            )
            continue

        quality = QUALITY_RE.search(line)
        if quality:
            euclidean_norm = float(quality.group(1))
            lift_norm = float(quality.group(2))
            gh = float(quality.group(3) or 0.0)
            approx = float(quality.group(4) or 0.0)
            if approx == 0.0 and gh > 0.0:
                approx = euclidean_norm / gh
            stats["quality_samples"] += 1
            stats["quality_final_euclidean_norm"] = euclidean_norm
            stats["quality_final_approx_factor"] = approx
            if (
                stats["quality_euclidean_norm"] == 0.0
                or euclidean_norm < stats["quality_euclidean_norm"]
            ):
                stats["quality_euclidean_norm"] = euclidean_norm
                stats["quality_lift_norm"] = lift_norm
                stats["quality_gh"] = gh
                if gh > 0.0:
                    stats["quality_challenge_bound"] = 1.05 * gh
                    stats["quality_challenge_pass"] = euclidean_norm <= 1.05 * gh
                stats["quality_approx_factor"] = approx

    flush_cuda()
    cuda_dp = stats["cuda_single_dp"] + stats["cuda_batch_dp"]
    cuda_sec = stats["cuda_single_sec"] + stats["cuda_batch_sec"]
    if cuda_sec > 0.0:
        stats["cuda_gdot_per_sec"] = cuda_dp / cuda_sec / 1.0e9
    return stats


def run_one(args, outdir, writer, jsonl, dim, lattice, mode, commit):
    command, mode_env = mode_command(args.app, lattice, mode, args.threads, args.log_level, args.seed)
    if args.print_quality:
        command.append("--print-final")
    env = os.environ.copy()
    env.update(mode_env)
    env["BGJ_SEED"] = str(args.seed)
    log_file = outdir / "logs" / f"d{dim}_{mode}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "kind": "bgj_app",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "seed": args.seed,
        "dim": dim,
        "mode": mode,
        "threads": args.threads,
        "timeout_sec": args.timeout_sec,
        "command": shlex.join(command),
        "env": json.dumps(mode_env, sort_keys=True),
        "lattice": str(lattice),
        "log_file": str(log_file),
    }
    row.update(empty_app_stats())

    if args.dry_run:
        row["returncode"] = ""
        row["timed_out"] = False
        row["elapsed_sec"] = 0.0
        writer.writerow(row)
        jsonl.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    start = time.monotonic()
    timed_out = False
    returncode = 0
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
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
            returncode = 124
            log.write(f"\n[TIMEOUT after {args.timeout_sec}s]\n")
    elapsed = time.monotonic() - start
    output = log_file.read_text(encoding="utf-8", errors="replace")
    row.update(parse_app_output(output))
    row["returncode"] = returncode
    row["timed_out"] = timed_out
    row["elapsed_sec"] = elapsed
    writer.writerow(row)
    jsonl.write(json.dumps(row, sort_keys=True) + "\n")
    jsonl.flush()
    return row


def run_suite(args):
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lattices").mkdir(parents=True, exist_ok=True)
    commit = repo_commit()

    dims = parse_csv_ints(args.dims) if args.dims else PRESET_DIMS[args.preset]
    modes = parse_csv_strings(args.modes) if args.modes is not None else PRESET_MODES[args.preset]

    for dim in dims:
        resolve_lattice(args, outdir, dim)

    if args.prepare_only:
        csv_path = outdir / "results.csv"
        jsonl_path = outdir / "results.jsonl"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore").writeheader()
        jsonl_path.write_text("", encoding="utf-8")
        return [], csv_path, jsonl_path

    csv_path = outdir / "results.csv"
    jsonl_path = outdir / "results.jsonl"
    rows = []
    suite_start = time.monotonic()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()
        for dim in dims:
            lattice = resolve_lattice(args, outdir, dim)
            for mode in modes:
                elapsed_budget = time.monotonic() - suite_start
                if elapsed_budget >= args.time_budget_sec:
                    print(f"stopping: suite time budget {args.time_budget_sec}s reached", file=sys.stderr)
                    return rows, csv_path, jsonl_path
                print(f"running dim={dim} mode={mode}", file=sys.stderr, flush=True)
                row = run_one(args, outdir, writer, jsonl, dim, lattice, mode, commit)
                rows.append(row)
                csv_file.flush()
                if args.stop_on_timeout and row.get("timed_out"):
                    print(f"stopping: dim={dim} mode={mode} hit timeout", file=sys.stderr)
                    return rows, csv_path, jsonl_path
                if args.stop_on_failure and int(row.get("returncode") or 0) != 0:
                    print(f"stopping: dim={dim} mode={mode} failed with returncode={row.get('returncode')}", file=sys.stderr)
                    return rows, csv_path, jsonl_path
    return rows, csv_path, jsonl_path


def print_summary(rows, csv_path, jsonl_path):
    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")
    print("kind,dim,mode,elapsed_sec,returncode,timeout,cuda_gdot_per_sec,cuda_single_dp,cuda_batch_dp,quality_norm,quality_bound,quality_approx,quality_pass")
    for row in rows:
        print(
            ",".join(
                [
                    str(row.get("kind", "")),
                    str(row.get("dim", "")),
                    str(row.get("mode", "")),
                    f"{float(row.get('elapsed_sec') or 0.0):.3f}",
                    str(row.get("returncode", "")),
                    str(row.get("timed_out", "")),
                    f"{float(row.get('cuda_gdot_per_sec') or 0.0):.6f}",
                    str(row.get("cuda_single_dp", "")),
                    str(row.get("cuda_batch_dp", "")),
                    f"{float(row.get('quality_euclidean_norm') or 0.0):.9g}",
                    f"{float(row.get('quality_challenge_bound') or 0.0):.9g}",
                    f"{float(row.get('quality_approx_factor') or 0.0):.9g}",
                    str(row.get("quality_challenge_pass", "")),
                ]
            )
        )


def parse_args(argv):
    default_out = ROOT / "bench" / "results" / time.strftime("seed42_%Y%m%d_%H%M%S", time.gmtime())
    parser = argparse.ArgumentParser(
        description="Run bounded deterministic BGJ/CUDA benchmarks with seed 42 by default."
    )
    parser.add_argument("--preset", choices=sorted(PRESET_DIMS), default="quick")
    parser.add_argument("--dims", help="comma-separated dimensions; overrides --preset")
    parser.add_argument("--modes", help="comma-separated modes; overrides --preset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-level", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int)
    parser.add_argument("--time-budget-sec", type=int)
    parser.add_argument("--app", type=Path, default=ROOT / "app" / "bgj_epi8")
    parser.add_argument("--preprocess-app", type=Path, default=ROOT / "app" / "lattice_preprocess")
    parser.add_argument("--lattice-template", help="existing lattice path template, e.g. tmp/L_{dim}")
    parser.add_argument("--svpchallenge-dir", type=Path, default=DEFAULT_SVPCHALLENGE_DIR)
    parser.add_argument("--svpchallenge-seed", type=int, default=0)
    parser.add_argument("--fetch-svpchallenge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-timeout-sec", type=int, default=120)
    parser.add_argument(
        "--validate-svpchallenge-generator",
        nargs="?",
        const=140,
        type=int,
        metavar="DIM",
        help="compare generator.php seed-0 output with a downloadable challenge example",
    )
    parser.add_argument("--allow-tmp-lattice", action="store_true")
    parser.add_argument("--synthetic-fallback", action="store_true")
    parser.add_argument("--preprocess-svp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess-timeout-sec", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--reuse-lattices", action="store_true")
    parser.add_argument("--print-quality", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-on-timeout", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    default_timeout, default_budget = PRESET_TIMEOUTS[args.preset]
    if args.timeout_sec is None:
        args.timeout_sec = default_timeout
    if args.time_budget_sec is None:
        args.time_budget_sec = default_budget
    args.svpchallenge_dir = args.svpchallenge_dir.resolve()
    return args


def main(argv):
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.app = args.app.resolve()
    if args.validate_svpchallenge_generator is not None:
        validate_svpchallenge_generator(args, args.validate_svpchallenge_generator)
    if not args.app.exists() and not args.dry_run:
        print(f"missing benchmark binary: {args.app}", file=sys.stderr)
        return 2
    if args.preprocess_svp and not args.preprocess_app.exists() and not args.dry_run:
        print(f"missing preprocessing binary: {args.preprocess_app}", file=sys.stderr)
        return 2
    rows, csv_path, jsonl_path = run_suite(args)
    print_summary(rows, csv_path, jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
