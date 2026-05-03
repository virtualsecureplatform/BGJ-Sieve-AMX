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
$ make
```

CUDA support can be enabled for the exact pair-search stage of `bgj1`. The
CPU implementation is still used for bucketing, lifting, insertion, and the
other sieve variants.

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

You can also force CUDA-assisted bucket search for a compiled-in `bgj1` run
with `BGJ_CUDA_SEARCH=1`. If CUDA is unavailable or an internal result buffer
overflows, the code falls back to the existing CPU search for that bucket. The
default result buffer holds `4194304` candidates per bucket and can be changed
with `BGJ_CUDA_MAX_RESULTS`.

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
`BGJ_CUDA_TENSOR_SAME_MIN_TILES=<n>` to tune the same-side threshold. A
shared-A positive/negative Tensor kernel and a wider 64x64 positive/negative
Tensor kernel are available for profiling with
`BGJ_CUDA_TENSOR_NP_SHARED_A=1` and `BGJ_CUDA_TENSOR_NP_WIDE=1`; both are off
by default on A100 because the lighter four-warp 16x16-tile path benchmarks
faster.
CUDA bucket search uses a nonblocking per-thread CUDA stream. By default, CUDA
search keeps the current pool vectors in a shared on-device cache and packs
bucket rows on-device; set `BGJ_CUDA_POOL_CACHE=0` to disable this and copy
each bucket from the host instead.
The default CUDA/BGJ build now instantiates `Pool_epi8_t<6>` and
`Pool_epi8_t<7>`, allowing non-LSH BGJ/CUDA paths to use 192- and
224-dimensional int8 pool vectors. The LSH and AMX paths remain capped by their
own template coverage.

A raw CUDA benchmark for small bucket-size sweeps lives in
`tests/bgj_cuda_raw_bench.cpp`. Build it against the CUDA library, then run it
directly or with Tensor disabled:

```bash
$ clang++ -O2 -g -std=c++11 -DHAVE_CUDA tests/bgj_cuda_raw_bench.cpp src/libllib.a \
    -Iinclude -Idep/ntl/include -Ldep/ntl/lib -lntl -Ldep/gmp/lib -lgmp -lm \
    -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart \
    -fopenmp=libomp -stdlib=libc++ -pthread -o /tmp/bgj_cuda_raw_bench
$ /tmp/bgj_cuda_raw_bench
$ BGJ_CUDA_TENSOR=0 /tmp/bgj_cuda_raw_bench
$ /tmp/bgj_cuda_raw_bench 160 8192 8192 100 1 1 8
```

For large instances, it's recommended to use [sparsepp](https://github.com/greg7mdp/sparsepp) to replace the default `std::unordered_set` used in the implementation of UidHashTable. This can be done by changing `USE_SPARSEPP` in `include/config.h` to 1 and manually placing the sparsepp headers into `dep/sparsepp/` before running make.

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
