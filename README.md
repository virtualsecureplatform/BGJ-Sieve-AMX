# BGJ-Sieve-AMX
A lattice algorithm library, including an efficient and memory-friendly implementation of the (practically) fastest BGJ sieve. It is based on:

> Ziyu Zhao, Jintai Ding and Bo-Yin Yang, Sieving with Streaming Memory Access

### Building the library

This library is optimized at a low level for modern Intel CPU architectures. The `avx512_vnni` instruction set is required, and the `amx_int8` instruction set is highly recommended for serious use. You can check whether your machine supports these instruction sets by running:

```bash
# Outputs 1 if supported
$ grep -q 'avx512_vnni' /proc/cpuinfo && echo "1" || echo "0"
# Outputs 1 if supported
$ grep -q 'amx_int8' /proc/cpuinfo && echo "1" || echo "0"
```

To compile the code, a recent version of `clang++` (e.g., `17.0.6`) with `libomp` and `libc++` should be properly installed. This is necessary because older compilers do not support `amx` instructions. Using `g++` is also possible but untested. We chose clang++ only because it seems to generate slightly smarter code. Now, build the library as follows:

```bash
# If gmp and ntl are not yet installed, you may install them using the shell scripts
# otherwise, change the include path in Makefile manually
$ ./install_gmp.sh
$ ./install_ntl.sh
$ git submodule update --init --recursive
$ make
```

Optional fplll support can replace the initial integer-matrix LLL step used by
`app/svp_solver` and `app/lattice_preprocess`. It can also accelerate the
regular LLL part of `tail_LLL`; the custom projected deep-LLL pass is still
used afterward. Build fplll from the checked-out submodule with the same
`clang++`/`libc++` ABI as this project:

```bash
$ git submodule update --init --recursive dep/fplll
$ sudo apt-get install -y autoconf automake libtool pkg-config libmpfr-dev
$ cd dep/fplll
$ ./autogen.sh
$ ./configure --prefix="$PWD/install" --with-gmp="$PWD/../gmp" --with-mpfr=/usr \
      CC=clang CXX=clang++ \
      CXXFLAGS="-O3 -g -march=native -stdlib=libc++ -fPIC" \
      LDFLAGS="-stdlib=libc++"
$ make -j"$(nproc)"
$ make install
$ cd ../../src && make
$ cd ../app && make
```

If `dep/fplll/install/include/fplll/fplll.h` exists, the Makefiles enable
`FPLLL=1` by default; pass `FPLLL=0` to disable it. When compiled with fplll,
fplll is the default initial LLL backend. Set `BGJ_LLL_BACKEND=ntl` to force
the previous NTL path. Set `BGJ_TAIL_LLL_BACKEND=custom` to force the previous
custom regular LLL inside `tail_LLL`. Set `BGJ_TAIL_DEEP_LLL_BACKEND=skip` to
skip the custom deep-LLL pass after a successful fplll tail LLL pass; this is
the default when fplll tail LLL succeeds. Set `BGJ_TAIL_DEEP_LLL_BACKEND=custom`
to restore the previous custom deep pass.

CUDA support can be enabled for BGJ sieve runs. In CUDA mode the A100 tuning
path uses GPU bucket search, GPU pool bucketing, and GPU candidate
materialization for BGJ1 by default. BGJ2 and BGJ3 reuse the same CUDA bucket
search backend for selected search phases, while bucket construction and the
insertion pass remain CPU-side.

```bash
$ make clean
$ make CUDA=1 CUDA_PATH=/usr/local/cuda CUDA_SMS=80
```

Set `CUDA_SMS` to the architectures you want to build for, for example
`CUDA_SMS="80 86 90"`. The CUDA build also accepts `NVCC` and `CUDA_CXX`
overrides when your CUDA toolkit needs a specific host compiler. At runtime,
`app/bgj_epi8` accepts `cuda` or `bgj1-cuda` as aliases for the CUDA-assisted
BGJ1 path:

```bash
$ ./bgj_epi8 L_100 cuda
```

`app/svp_tool` accepts `--cuda` for the BGJ sieve phases inside the SVP pump flow.
That mode is conservative by default: it keeps the CPU bucket density unless a
bucket target/alpha is explicitly provided, skips CUDA search below the
configured dot-product threshold, and disables CUDA candidate materialization
unless `BGJ_CUDA_MATERIALIZE=1` is set. This avoids the small-bucket launch and
materialization overheads that make low-dimensional pump stages slower than the
CPU path.
`app/svp_solver` also accepts `--cuda`; it enables the same
`BGJ_SVP_CUDA=1` hook used by the solver internals and enables CUDA LSH search
unless `BGJ_CUDA_LSH_SEARCH=0` is set. For the `120t95` schedule, the default
final LSH policy is an automatic quality rescue: after the normal schedule it
compares the best basis row and the best recorded LSH candidate against
`BGJ_120T95_FINAL_LSH_TARGET_FACTOR` times GH, default `0.97`, and runs the
standalone final LSH pass only if that candidate is still worse. Set
`BGJ_SVP_FINAL_LSH=0` to disable the rescue or `BGJ_SVP_FINAL_LSH=force` to run
it unconditionally. `BGJ_120T95_FINAL_LSH_TARGET=<length>` sets an absolute
target. The final full-dimensional `lsh_pump_92` also stops early in auto mode
once it records a candidate at the same target, skipping the trailing local pump
and rescue; override that with `BGJ_120T95_LSH_PUMP_STOP_LENGTH=<length>`.
If that pump misses, `BGJ_120T95_LSH_RETRY=<n>` can run one or more
single-insert late-LSH probes before final rescue; it defaults to `0` because
the final rescue is now quality-driven. Tune
`BGJ_120T95_LSH_RETRY_NI`, `BGJ_120T95_LSH_RETRY_MSD`,
`BGJ_120T95_LSH_RETRY_F`, and `BGJ_120T95_LSH_RETRY_QRATIO`.
When auto mode decides the final rescue is needed, the trailing
`local_pump_85_15_120_b` remains enabled by default to preserve the validated
rescue trajectory; set `BGJ_120T95_LOCAL_PUMP_B_BEFORE_RESCUE=0` to skip it
for experiments.
The final rescue delays lift probes until `BGJ_120T95_FINAL_LSH_LIFT_MARGIN`
below the final MSD, default `12` for the validated SVP-120 rescue trajectory;
use `BGJ_120T95_FINAL_LSH_LIFT_START_CSD=<dim>` for an absolute start point.
Smaller margins reduce rescue lift probes but can miss the best lifted vector;
larger margins probe earlier and may shorten later BGJ3 rescue work.
In auto mode, `BGJ_120T95_FINAL_LSH_ATTEMPTS` defaults to `2`, so a rescue
pass that still misses the GH-based quality target is followed by one more
rescue pass. Set it to `1` to restore the old single-shot behavior.
`BGJ_120T95_FINAL_LSH_FAST_FIRST=1` tries a cheaper first rescue pass
(`msd=92`) before falling back to the full final LSH pass; it is off by default
because the cheap pass can perturb the full rescue trajectory.
`BGJ_120T95_FINAL_LSH_STOP_LENGTH=<length>` overrides the rescue early-stop
threshold.
Plain `--cuda` uses one CUDA execution device unless `BGJ_CUDA_DEVICES` or
`BGJ_CUDA_NUM_DEVICES` is set. The SVP-120 quality-rescue default is validated
on that single-device trajectory. CUDA BGJ2 first-level search uses the staged
schedule by default so candidate consumption order is stable; set
`BGJ_CUDA_BGJ2_SEARCH0_STAGED=0` to restore the older parallel bucket-pull path
for experiments.
For BGJ2, CUDA search is enabled for first-level reuse buckets by default.
Second-level subbucket search is available but off by default for SVP quality.
`BGJ_CUDA_BGJ2_SEARCH=0` disables BGJ2 search offload,
`BGJ_CUDA_BGJ2_SEARCH0=0` disables only the first-level reuse buckets,
`BGJ_CUDA_BGJ2_SEARCH1=1` enables the second-level subbucket search, and
`BGJ_CUDA_BGJ2_MIN_DOTS` sets the BGJ2-only search threshold.
BGJ2-only search batching can be tested with `BGJ_CUDA_BGJ2_BATCH=1` or
`BGJ_CUDA_BGJ2_BATCH_SIZE=<n>` without changing the BGJ1 batch setting. In a
May 2026 A100/GPU1 dim-96 direct BGJ2 run, enabling second-level CUDA search
reduced wall time from `240.54s` to `181.81s` with the same final norm and
approximation factor.
For BGJ3, CUDA search is enabled by default for `search0`, `search1`, and
`search2` buckets whose dot-product work crosses the configured threshold.
Use `BGJ_CUDA_BGJ3_SEARCH=0` to disable all BGJ3 search offload, or
`BGJ_CUDA_BGJ3_SEARCH0=0`, `BGJ_CUDA_BGJ3_SEARCH1=0`, and
`BGJ_CUDA_BGJ3_SEARCH2=0` to disable individual phases.
`BGJ_CUDA_BGJ3_MIN_DOTS` sets the BGJ3-only threshold; the default follows
`BGJ_CUDA_BATCH_MIN_DOTS`. Very low thresholds, such as `1`, are useful for
smoke-testing that BGJ3 calls CUDA, but are not a tuned performance setting.
BGJ3 `search2` uses the full fused CUDA batch by default when CUDA search is
enabled. Set `BGJ_CUDA_BGJ3_SEARCH2_FULL_FUSED=0` to disable this default, or
override the batch size with `BGJ_CUDA_BGJ3_BATCH_SIZE=<n>`.
`BGJ_CUDA_BGJ3_BATCH=1` enables the older non-fused batch path when the
full-fused path is disabled, and `BGJ_CUDA_BGJ3_MIN_BATCH=<n>` controls the
minimum eligible buckets needed before using that older batch kernel.
`BGJ_CUDA_BGJ3_SEARCH2_STAGED=1` enables ordered staged result consumption for
`search2` batches. Its batch-admission gate is controlled separately by
`BGJ_CUDA_BGJ3_SEARCH2_STAGED_MIN_DOTS`, defaulting to `1048576`, so staged
experiments can admit small ordered batches without lowering the normal BGJ3
single-bucket CUDA threshold.
`BGJ_CUDA_BGJ3_SEARCH2_NP_FUSED=1` enables an experimental fused NP-only CUDA
batch for BGJ3 `search2`. It collects many bucket2 mixed p/n searches into one
launch, consumes results in bucket order, then runs the existing CPU pp/nn tail.
`BGJ_CUDA_BGJ3_SEARCH2_FULL_FUSED` extends that fused batch to include the
same-sign pp/nn searches as well and is on by default; bucket order is still
preserved, and buckets fall back to the ordinary CPU path on CUDA failure,
result overflow, or result consume failure.
`BGJ_CUDA_BGJ3_SEARCH2_NP_FUSED_MIN_DOTS` controls the fused batch gate and
defaults to `1048576`.
`BGJ_CUDA_BGJ3_SEARCH2_NP_FUSED_MAX_RESULTS` caps the per-bucket result slab
used by the fused path. By default this path uses at most `262144` results per
bucket, while ordinary CUDA search still honors `BGJ_CUDA_MAX_RESULTS`.
For BGJ3 `search2` CPU fallback diagnosis, set
`BGJ_CUDA_BGJ3_SEARCH2_CPU_PROFILE=1` together with `BGJ_CUDA_BGJ3_PROFILE=1`;
the `bgj3_cuda_profile` line then reports split fallback time for `cred`,
`np`, `pp`, and `nn`.

For reproducible CUDA/BGJ profiling, pass a fixed sampler seed as the fifth
positional argument, with `-s/--seed`, or with `BGJ_SEED`:

```bash
$ ./bgj_epi8 L_100 cuda 1 2 42
$ BGJ_SEED=42 ./bgj_epi8 L_100 cuda 1 2
```

The seed controls the library sampler and the internal randomized choices used
by sieving. `svp_tool -s` and `svp_solver -s` use the same sampler seed path.

You can also force CUDA-assisted bucket search for a compiled-in `bgj1` run
with `BGJ_CUDA_SEARCH=1`. If CUDA is unavailable, the code falls back to the
existing CPU search. The default result buffer holds `16777216` candidates per
bucket and can be changed with `BGJ_CUDA_MAX_RESULTS`.

The CUDA BGJ1 path enables pool-bucketing offload by default. It preserves the
current threshold-based bucket
membership rule, writes the same positive/negative dot-product records used by
the CPU search path, and falls back to CPU bucketing if the CUDA output buffer
overflows. Tune the temporary output buffer with `BGJ_CUDA_BUCKET_MARGIN=<n>`
or cap it with `BGJ_CUDA_BUCKET_MAX_ENTRIES=<n>`. Set `BGJ_CUDA_BUCKET=0` to
force CPU bucketing. On A100, CUDA bucketing
defaults to the append-based INT8 Tensor Core kernel for full 16x16
center/vector tiles because it is the fastest current path on SVP100 timing
runs. Set `BGJ_CUDA_BUCKET_TENSOR=0` to force scalar `dp4a` append kernels.
The deterministic two-pass `dp4a` path remains available with
`BGJ_CUDA_BUCKET_DETERMINISTIC=1`; it emits each bucket as positive ids in
ascending order followed by negative ids in ascending order, but is slower in
the current A100 benchmarks. On A100, scalar append bucketing uses block-local
compaction by default to reduce global append atomics; set
`BGJ_CUDA_BUCKET_BLOCK_APPEND=0` to force the older per-entry append path.

On A100-class GPUs, CUDA search uses experimental INT8 Tensor Core paths for
full 16x16 bucket tiles by default. Tensor kernels process four independent
tiles per CTA to keep more warps resident on A100. The positive/negative tile
path defaults to Tensor only after at least 512 full tiles are available;
same-side positive/positive and negative/negative Tensor tiles run only for
larger bucket sides. Fringe pairs and smaller buckets still use the CUDA
`dp4a` kernel. Tensor paths prepackage int8 WMMA fragments by default so
repeated tile loads use the A100 read-only cache. Set
`BGJ_CUDA_TENSOR=0` to disable Tensor Cores,
`BGJ_CUDA_TENSOR_REORDER=0` to disable the fragment prepack path,
`BGJ_CUDA_TENSOR_NP_MIN_TILES=<n>` to tune the positive/negative Tensor
threshold, `BGJ_CUDA_TENSOR_SAME=0` to disable only same-side Tensor tiles, or
`BGJ_CUDA_TENSOR_SAME_MIN_TILES=<n>` to tune the same-side threshold. A wider
64x64 positive/negative Tensor kernel is the default on sm80/A100; set
`BGJ_CUDA_TENSOR_NP_WIDE=0` to use the single-tile path. Shared-A and 32x256
multi-fragment positive/negative Tensor kernels are available for profiling
with `BGJ_CUDA_TENSOR_NP_SHARED_A=1` and `BGJ_CUDA_TENSOR_NP_MULTI=1`.
CUDA bucket search uses a nonblocking per-thread CUDA stream. By default, CUDA
search keeps the current pool vectors in a shared on-device cache and packs
bucket rows on-device; set `BGJ_CUDA_POOL_CACHE=0` to disable this and copy
each bucket from the host instead. CUDA BGJ1 uses two host worker threads by
default at dimension 80 and above so bucketing and materialization can use more
CPU parallelism; smaller dimensions keep the single-thread path. Set
`BGJ_CUDA_HOST_THREADS=<n>` to force a different count. CUDA bucket search uses
the same host count by default so independent bucket streams can overlap; set
`BGJ_CUDA_SEARCH_THREADS=1` to keep result consumption single-threaded when
debugging scheduling-sensitive runs. The raw pool interface also has an
experimental batched entry point, `bgj_cuda_search_bucket_pool_batch_raw`,
which submits several buckets on separate CUDA scratch streams and synchronizes
once for counts before copying only the valid result ranges. The BGJ1 CUDA
sieve loop leaves batching off by default on A100 because single-bucket
submission lets the BGJ1 early-stop logic react sooner on the tuned SVP
challenge runs. Set `BGJ_CUDA_BATCH=1` to enable batching,
`BGJ_CUDA_BATCH_SIZE=<n>` to tune the group size, or
`BGJ_CUDA_BATCH_MIN_DOTS=<n>` to tune the size threshold. With BGJ log level
`2` or higher, the BGJ1 profile prints CUDA single-bucket, batched, cred, and
fallback bucket counts, dot counts, and timings.
CUDA single-bucket search overlaps pair search with the CPU `_search_cred` pass
by default. The pair kernels read raw center-dot values directly and the host
waits only for a dot-copy CUDA event before running `_search_cred`. Set
`BGJ_CUDA_OVERLAP_CRED=0` to disable this path when comparing schedules.
Multi-GPU execution is opt-in. Set `BGJ_CUDA_DEVICES=0,1` to use two visible
runtime devices; `BGJ_CUDA_NUM_DEVICES=2` is a shorthand for devices `0,1`.
Explicit multi-GPU enables the rest of the stable schedule by default: the
primary host thread is pinned to a CUDA device and large single buckets are
split across GPUs. Each GPU keeps its own pool cache and scratch streams. CUDA
sieve wrappers raise the host search thread count to at least the active GPU
count unless
`BGJ_CUDA_HOST_THREADS` is set explicitly, so two-device runs have enough host
work to feed both GPUs. Set `BGJ_CUDA_STABLE_MULTI_GPU=0` to restore the older
opportunistic schedule for experiments, including the older BGJ2 search0
default. `BGJ_CUDA_SINGLE_BUCKET_SPLIT=0` disables large-bucket splitting;
under the stable multi-GPU default the split threshold is `16000000` pair dots
and can be changed with `BGJ_CUDA_SINGLE_BUCKET_SPLIT_MIN_DOTS=<n>`. The
SVP-100 benchmark harness accepts `--cuda-devices 0,1`, which records the
setting in the result environment JSON.
The `app/bgj_epi8 ... cuda` entry point now follows the stronger CPU `bgjf`
progression: CUDA BGJ1 for the initial and low-dimensional stages, then BGJ2
and BGJ3 with CUDA search offload above the same thresholds as `bgjf`. The old
BGJ1-only CUDA path is still available as `bgj1-cuda`; the benchmark harness
selects it with modes prefixed by `cuda-bgj1`. The hybrid path keeps CUDA
candidate materialization enabled for BGJ2/BGJ3 inserts.
Do not enable search batching by default without rebenchmarking: a May 2026
A100/GPU1 SVP100 seed-42 challenge run with a 600s timeout regressed from
`281.15s` total / `102.80s` search in the default single-bucket mode to
`414.83s` total / `211.26s` search with `cuda-batch4`, because batching delays
the BGJ1 early-stop check and creates many more solution records. In the same
run, `BGJ_CUDA_TENSOR_REORDER=0` was only marginally faster
(`280.69s` total / `101.66s` search) and is not enough evidence to change the
default fragment prepack path. The benchmark harness accepts profiling modes
such as `cuda-no-reorder`, `cuda-wide-off`, `cuda-np-multi`,
`cuda-shared-a`, `cuda-np-min1024`, and `cuda-same-min64`.
For A100 tuning, BGJ1 bucket density can be changed without rebuilding:
`BGJ1_EPI8_BUCKET_ALPHA=<x>` sets the alpha directly, while
`BGJ1_EPI8_BUCKET_TARGET_SIZE=<n>` derives an alpha for the current pool size
and sieving dimension. Lower alpha means larger buckets, which can feed the
Tensor Core path better but may also increase solution materialization and
insertion cost. CUDA BGJ1 runs at dimension 40 or larger default to target
`12288` unless an alpha or target is explicitly set. Benchmark modes such as
`cuda-target12288` set this target through `bench/bgj_cuda_seed42.py`; on the
current A100 dim70/dim80 seed-42 challenge runs, this higher-density setting is
best for dim80 and still leaves dim70 well ahead of the G6K comparison.
Dense buckets can produce many CUDA solution records; `BGJ_CUDA_MAX_RESULTS`
sets the per-bucket result capacity and defaults to `1048576` for the A100
tuning path. On overflow the CUDA path now consumes the capped result prefix and
continues, matching G6K-GPU-Tensor's bounded queue behavior; set
`BGJ_CUDA_OVERFLOW_FALLBACK=1` to restore CPU fallback on overflow. CUDA result
records are consumed in device result order by default, matching G6K's queue
path and avoiding a full host-side sort of dense result buffers; set
`BGJ_CUDA_SORT_RESULTS=1` to restore the older CPU-order approximation for
debugging. `BGJ_CUDA_DETERMINISTIC_RESULTS=1` enables an experimental two-pass
GPU result emitter that writes records in deterministic phase/rank order
without a host sort. It is currently a profiling path rather than a default:
the scalar A100 implementation avoids the CPU sort but also skips the Tensor
Core search kernels.
The benchmark harness also accepts modes such as `cuda-maxres1048576` and
`cuda-maxres4194304` and can record final vector quality with `--print-quality`.
Use `--require-quality` on `app/bgj_epi8` or `bench/bgj_cuda_seed42.py` to make
the final lifted vector a hard SVP challenge gate: the app exits with code `3`
unless `norm <= gamma * GH`. The default is the SVP challenge factor
`gamma = 1.05`; override it with `--quality-gamma <gamma>` when testing tighter
or looser criteria.
Before the hybrid `bgjf` CUDA entry point, the BGJ1-only SVP-100 seed-0
challenge lattice preprocessed by `app/lattice_preprocess` and run on A100/GPU1
with seed 42 and a 600s per-mode timeout had these results:
`cuda-bgj1-maxres1048576` took `214.67s` and reported Euclidean norm
`2860.50415` with approximation factor `1.12639278`;
`cuda-bgj1-maxres4194304` took `239.46s` with norm `2895.50548` and factor
`1.14017540`; the previous uncapped BGJ1-only default took `297.26s` with norm
`2866.45583` and factor `1.12873640`. The challenge bound for that run is
`1.05 * GH = 2666.50267`, so none of those app-level SVP-100 BGJ1-only runs is
a valid SVP challenge solution yet. On smaller seed-0 challenge cases, the
hybrid `cuda` schedule did not improve final quality over `cuda-bgj1`: dim 82
found norm `2302.66823` with factor `0.995665088` in `17.63s` versus `11.15s`
for BGJ1-only, and dim 92 found norm `2440.24364` with factor `0.996808924` in
`226.81s` versus `55.91s` for BGJ1-only. This suggests BGJ2/BGJ3 search
offload should be justified on harder cases where the stronger schedule changes
quality, not by these small passing instances.
CUDA BGJ1 enables the CUDA candidate materializer by default. It chunks
solution records, uses signed INT8 cuBLAS GEMM for coefficient reconstruction,
and defaults to the exact CUDA reconstruction kernel on A100. It keeps the
basis resident on the GPU across repeated calls when the hashed
`_b_dual`/`_b_local` content is unchanged and copies device output directly to
the insertion buffers by default. Set `BGJ_CUDA_MATERIALIZE=0` to force CPU
candidate materialization. Set
`BGJ_CUDA_MATERIALIZE_BASIS_CACHE=0` when isolating basis upload costs,
`BGJ_CUDA_MATERIALIZE_PINNED_HOST=1` to re-enable the older pinned-host staging
path, `BGJ_CUDA_MATERIALIZE_SGEMM=1` to use the SGEMM reconstruction path, or
`BGJ_CUDA_MATERIALIZE_FUSED_COEFF=1` to test fused int32 coefficient conversion.
Tune the cuBLAS chunk size with `BGJ_CUDA_MATERIALIZE_CHUNK=<n>` and the exact
finish block size with `BGJ_CUDA_MATERIALIZE_THREADS=<32|64|128|256>` when
profiling. Set `BGJ_CUDA_MATERIALIZE_PHASE_PROFILE=1` to print CUDA phase
timings for pool cache, basis upload, descriptor copy, vector build, GEMM,
coefficient conversion, reconstruction, and output copy.
For dimensions 96 and higher, the A100 path defaults to staged output copying:
the GPU keeps candidate vectors resident, copies only norms and sums first, and
then gathers host vectors only for candidates the insertion pass will keep. Set
`BGJ_CUDA_MATERIALIZE_STAGED=0` to force the older full-copy path, or `1` to
force staged copying at smaller dimensions. Do not merge the first insertion
selection scan into the staged gather path without rebenchmarking: a May 2026
A100/GPU1 SVP96 seed-42 test that cached final insertion destinations during
staged selection regressed from `136.38s` total / `45.79s` insert to `149.50s`
total / `57.03s` insert, so the simpler current two-scan insertion path remains
faster.
A fused one-kernel small-batch materializer is also available with
`BGJ_CUDA_MATERIALIZE_FUSED=1` and capped by
`BGJ_CUDA_MATERIALIZE_FUSED_MAX=<n>`, but it is not the default on A100 because
cached cuBLAS is usually faster in the current microbenchmarks. Set
`BGJ_CUDA_MATERIALIZE_HYBRID=1` to split a materialization batch between GPU
and CPU; tune the GPU share with `BGJ_CUDA_MATERIALIZE_GPU_PERCENT=<0..100>` or
an exact `BGJ_CUDA_MATERIALIZE_GPU_COUNT=<n>`. In CUDA builds,
the shared descriptor materializer uses 8 CPU threads by default for CUDA BGJ1
runs. Set `BGJ_CPU_MATERIALIZE_THREADS=<n>` to tune it, or `0` to use the old
single-thread materialization path. This remains CPU-side candidate
materialization only; CUDA search and bucketing timings are reported
separately.
Current A100/GPU1 seed-42 smoke checks against SVP challenge seed-0 inputs show
the expected crossover: dim50 and dim60 are still CPU-faster because launch and
setup costs dominate, dim70 is faster with CUDA (`2.07s` vs `2.62s`), and dim80
is substantially faster with CUDA (about `7-9s` vs `29.68s`). In those
dim70/dim80 runs the final Euclidean norm and approximation factor matched the
CPU result.
CUDA BGJ1 batches UID erases during insertion by default. This reduces lock
traffic in the current A100 SVP96 timing runs and can be disabled with
`BGJ_INSERT_BATCH_UID_ERASE=0`. Set `BGJ_INSERT_PHASE_PROFILE=1` to print the
insert scan, UID erase, batch erase, copy, and compact phase timings.
The default CUDA/BGJ build now instantiates `Pool_epi8_t<6>` and
`Pool_epi8_t<7>`, allowing non-LSH BGJ/CUDA paths to use 192- and
224-dimensional int8 pool vectors. The LSH and AMX paths remain capped by their
own template coverage.

Raw SVP-challenge q-ary bases should be preprocessed before direct `bgj_epi8`
runs. The challenge files contain very large unreduced entries, while the int8
BGJ path expects a reduced/local basis. Use `app/lattice_preprocess` to run
exact NTL LLL and write a reduced NTL-format basis:

```bash
$ app/lattice_preprocess ../G6K-GPU-Tensor/svpchallenge/svpchallenge-dim-080-seed-00.txt /tmp/svp80-lll.txt
$ app/bgj_epi8 /tmp/svp80-lll.txt cuda 1 2 42
```

A raw CUDA benchmark for small bucket-size sweeps lives in
`tests/bgj_cuda_raw_bench.cpp`. Build it against the CUDA library, then run it
directly or with Tensor disabled:

```bash
$ clang++ -O2 -g -std=c++11 -DHAVE_CUDA tests/bgj_cuda_raw_bench.cpp src/libllib.a \
    -Iinclude -Idep/ntl/include -Ldep/ntl/lib -lntl -Ldep/gmp/lib -lgmp -lm \
    -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas \
    -fopenmp=libomp -stdlib=libc++ -pthread -o /tmp/bgj_cuda_raw_bench
$ /tmp/bgj_cuda_raw_bench
$ BGJ_CUDA_TENSOR=0 /tmp/bgj_cuda_raw_bench
$ /tmp/bgj_cuda_raw_bench 160 8192 8192 100 1 1 8
$ /tmp/bgj_cuda_raw_bench 224 8192 8192 50 1 1 1 8
```

The raw CUDA bucketer has a small oracle test:

```bash
$ clang++ -O2 -g -std=c++11 -DHAVE_CUDA tests/bgj_cuda_bucket_test.cpp src/libllib.a \
    -Iinclude -Idep/ntl/include -Ldep/ntl/lib -lntl -Ldep/gmp/lib -lgmp -lm \
    -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas \
    -fopenmp=libomp -stdlib=libc++ -pthread -o /tmp/bgj_cuda_bucket_test
$ /tmp/bgj_cuda_bucket_test
```

The raw CUDA materializer has a small correctness test:

```bash
$ clang++ -O2 -g -std=c++11 -DHAVE_CUDA tests/bgj_cuda_materialize_test.cpp src/libllib.a \
    -Iinclude -Idep/ntl/include -Ldep/ntl/lib -lntl -Ldep/gmp/lib -lgmp -lm \
    -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas \
    -fopenmp=libomp -stdlib=libc++ -pthread -o /tmp/bgj_cuda_materialize_test
$ /tmp/bgj_cuda_materialize_test
```

There is also a standalone materializer microbenchmark. It compares the raw
CUDA materializer against a scalar host implementation of the same arithmetic;
use end-to-end BGJ insert timings to compare against the library's optimized
AVX2 CPU materializer.

```bash
$ clang++ -O3 -g -std=c++11 -DHAVE_CUDA tests/bgj_cuda_materialize_bench.cpp src/libllib.a \
    -Iinclude -Idep/ntl/include -Ldep/ntl/lib -lntl -Ldep/gmp/lib -lgmp -lm \
    -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas \
    -fopenmp=libomp -stdlib=libc++ -pthread -o /tmp/bgj_cuda_materialize_bench
$ /tmp/bgj_cuda_materialize_bench 128 16384 3 8192 1
```

To compare 256-bit VNNI against an experimental 512-bit VNNI/FMA CPU
materializer kernel on the local CPU, use:

```bash
$ clang++ -O3 -g -std=c++11 tests/bgj_cpu_materialize_avx512_bench.cpp \
    -Iinclude -fopenmp=libomp -stdlib=libc++ -pthread -march=native \
    -o /tmp/bgj_cpu_materialize_avx512_bench
$ /tmp/bgj_cpu_materialize_avx512_bench 224 16384 5 8192
```

For end-to-end A100 tuning on seed-42 SVP-challenge ladders, use:

```bash
$ bench/bgj_cuda_seed42.py --preset ladder --stop-on-timeout --stop-on-failure \
    --timeout-sec 3600 --time-budget-sec 28800
```

To reproduce the official dim-100 seed-46937 solver case from
`tmp/solutions.txt`, use the dedicated `svp_solver` target. It generates the
raw SVP challenge basis with the old NTL-compatible official generator when the
basis is missing, runs `SVPALGO_100T90`, and records runtime plus the SVP
challenge quality check under `bench/results/`.

```bash
$ CUDA_VISIBLE_DEVICES=1 bench/svp_solver_100t90_seed46937.py --mode cuda
$ CUDA_VISIBLE_DEVICES=1 bench/svp_solver_100t90_seed46937.py --mode both
$ CUDA_VISIBLE_DEVICES=1 bench/svp_solver_100t90_seed46937.py --mode cuda --profile
```

The harness uses the official SVP-challenge seed-0 generator when an instance
is missing from `../G6K-GPU-Tensor/svpchallenge`, preprocesses challenge
instances with `app/lattice_preprocess`, writes live logs, and records CSV/JSONL
summaries under `bench/results/`. It no longer falls back to `tmp/L_<dim>` or
synthetic lattices by default; use `--allow-tmp-lattice` or
`--synthetic-fallback` only when deliberately benchmarking non-challenge inputs.
Use `--validate-svpchallenge-generator` to compare the online generator against
the downloadable seed-0 dim-140 example, and `--prepare-only` to fetch and
preprocess lattices without launching sieving jobs.

For large instances, the default Makefile build uses
[gtl](https://github.com/greg7mdp/gtl) `gtl::flat_hash_set` for
`UidHashTable`. The UID table is insert-heavy during CUDA sieving, and the gtl
backend avoids the `std::unordered_set` rehash/pointer-chasing bottleneck while
also improving larger SVP-challenge CUDA runs in current A100 tests. Build with
`UID_BACKEND=sparsepp` to use the older vendored
[sparsepp](https://github.com/greg7mdp/sparsepp) backend instead. The UID table
pre-reserves aggressively by default; tune this with `BGJ_UID_RESERVE_FACTOR`
(default `8.0`) and cap it with `BGJ_UID_RESERVE_MAX` entries (default
`268435456`, `0` means uncapped).

For large instances, you may need to modify the values of `AMX_MAX_NTHREADS` in line 16 of `include/bgj_amx.h` and `MAX_NTHREADS` in line 42 of `include/bgj_epi8.h`, then recompile the code. The default value for both is set to 112.

### How to use

After compiling, the file `app/svp_tool` will be generated, almost all the algorithms we implemented can be call via this command line tool, here are some examples about how to use it.

> The input basis should be in "NTL format" and full rank. Because we use `NTL::quad_float` to store the entries, the input entries should not be too large. It seems to work well if all input integer entries are in the range [-2^30, 2^30]. 

```bash
$ cd tmp && cp ../app/svp_tool .
$ ./svp_tool --help   # show the help message
Usage: ./svp_tool <inputfile> [options]
Options:
  -h, --help			Show this help message
  -l, --ind_l			Left index of local processing						# default = 0
  -r, --ind_r			Right index of local processing						# default = dimension of input
  -s, --seed			Seed for random number generator
  -t, --threads			Number of threads									# should <= 112
  -v, --verbose			Verbosity level, default = 3
  -f, --final			Final run											# print short vectors directly  
  -lsh, --lsh			Use lsh lifting and specify qratio and ext_qratio	# need ind_r - msd > ind_l + 24
  -ssd, --ssd			Start sieving dimension
  -msd, --msd			Maximal sieving dimension
  -esd, --esd			Extended sieving dimension
  -ds, --down_sieve		Down sieve											# enabled by default
  -amx, --amx			Use amx

# Run pump and store the output basis to a file ("L_112r" by default). Sieving context [20, 110], expected dim for free = 22, down sieve is enabled by default
$ ./svp_tool L_112 --msd 90 --threads 8

# Run pump on the 108-dimensional local projected lattice, sieving dimension = 100, expected dim for free = 8. A size reduction and an LLL will be applied to the full basis after the local processing is done. The output file name is the same as the input if the option "-sr" is set.
$ ./svp_tool L_112 --ind_l 2 --ind_r 110 --msd 100 --threads 64 -sr
```

### Reproducing experiments of the paper for the command line

#### Performance of bgj1,bgj2 and bgj3 (Sec 4.1, Fig.1)

You can try `bgj1`, `bgj2`, and `bgj3` with `app/bgj_epi8`. 

```bash
$ cd tmp && cp ../app/bgj_epi8 .
# Run 100 dimensional left progressive bgj1/bgj2/bgj3 sieve
$ ./bgj_epi8 L_100 bgj1
$ ./bgj_epi8 L_100 bgj2
$ ./bgj_epi8 L_100 bgj3
```

#### Reproducing the 179-dimensional challenge (Sec 6.4)

We provide the shell script we used for the 179-dimensional challenge in `tmp/svpchallenge179.sh`. Using this script, we found a solution in approximately 2 weeks (not all the commands used 112 threads) on a dual Intel Xeon Platinum 8479 server with 1TB RAM. The maximum sieving dimension was 147. We also include all other Darmstadt SVP challenge solutions we found in `tmp/solutions.txt`.

```bash
$ cd tmp && cp ../app/svp_tool .
$ nohup ./svpchallenge179.sh L_179r >> log &
```

#### Gate count of the 100-dimensional sieve (Sec 7)

```bash
$ cd tmp && cp ../app/bgj_epi8 .
$ nohup time ./bgj_epi8 L_100 bgjf 1 1 >> gate_count.log &
```
