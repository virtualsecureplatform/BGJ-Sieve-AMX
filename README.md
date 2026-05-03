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
