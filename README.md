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

CUDA support can be enabled for the exact pair-search stage of `bgj1`. The
CPU implementation is still used by default for bucketing, lifting, insertion,
and the other sieve variants.

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

For reproducible CUDA/BGJ profiling, pass a fixed sampler seed as the fifth
positional argument, with `-s/--seed`, or with `BGJ_SEED`:

```bash
$ ./bgj_epi8 L_100 cuda 1 2 42
$ BGJ_SEED=42 ./bgj_epi8 L_100 cuda 1 2
```

The seed controls the library sampler and the internal randomized choices used
by sieving. `svp_tool -s` and `svp_solver -s` use the same sampler seed path.

You can also force CUDA-assisted bucket search for a compiled-in `bgj1` run
with `BGJ_CUDA_SEARCH=1`. If CUDA is unavailable or an internal result buffer
overflows, the code falls back to the existing CPU search for that bucket. The
default result buffer holds `4194304` candidates per bucket and can be changed
with `BGJ_CUDA_MAX_RESULTS`.

An experimental BGJ1 pool-bucketing offload is available with
`BGJ_CUDA_BUCKET=1`. It preserves the current threshold-based bucket
membership rule, writes the same positive/negative dot-product records used by
the CPU search path, and falls back to CPU bucketing if the CUDA output buffer
overflows. Tune the temporary output buffer with `BGJ_CUDA_BUCKET_MARGIN=<n>`
or cap it with `BGJ_CUDA_BUCKET_MAX_ENTRIES=<n>`. An A100 INT8 Tensor Core
bucketing kernel for full 16x16 center/vector tiles is available for profiling
with `BGJ_CUDA_BUCKET_TENSOR=1`, but the scalar `dp4a` bucketing kernel is the
default because it is faster in the current SVP70/SVP80 timing runs.
On A100, scalar bucketing uses block-local compaction by default to reduce
global append atomics; set `BGJ_CUDA_BUCKET_BLOCK_APPEND=0` to force the older
per-entry append path.

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
sets the per-bucket result capacity and defaults to `16777216` for the A100
tuning path. On overflow the CUDA path now consumes the capped result prefix and
continues, matching G6K-GPU-Tensor's bounded queue behavior; set
`BGJ_CUDA_OVERFLOW_FALLBACK=1` to restore CPU fallback on overflow. CUDA result
records are consumed in device result order by default, matching G6K's queue
path and avoiding a full host-side sort of dense result buffers; set
`BGJ_CUDA_SORT_RESULTS=1` to restore the older CPU-order approximation for
debugging.
An experimental CUDA candidate materializer is available with
`BGJ_CUDA_MATERIALIZE=1`. It chunks solution records, uses signed INT8 cuBLAS
GEMM for coefficient reconstruction, and defaults to the exact CUDA
reconstruction kernel on A100. It keeps the basis resident on the GPU across
repeated calls when the hashed `_b_dual`/`_b_local` content is unchanged and
copies device output directly to the insertion buffers by default. Set
`BGJ_CUDA_MATERIALIZE_BASIS_CACHE=0` when isolating basis upload costs,
`BGJ_CUDA_MATERIALIZE_PINNED_HOST=1` to re-enable the older pinned-host staging
path, `BGJ_CUDA_MATERIALIZE_SGEMM=1` to use the SGEMM reconstruction path, or
`BGJ_CUDA_MATERIALIZE_FUSED_COEFF=1` to test fused int32 coefficient conversion.
Tune the cuBLAS chunk size with `BGJ_CUDA_MATERIALIZE_CHUNK=<n>` and the exact
finish block size with `BGJ_CUDA_MATERIALIZE_THREADS=<32|64|128|256>` when
profiling. Set `BGJ_CUDA_MATERIALIZE_PHASE_PROFILE=1` to print CUDA phase
timings for pool cache, basis upload, descriptor copy, vector build, GEMM,
coefficient conversion, reconstruction, and output copy.
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
