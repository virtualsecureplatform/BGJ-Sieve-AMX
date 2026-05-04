OBJ = config.o pool_epi8.o bucket_epi8.o bgj_epi8.o bgj_cuda.o 3Sieve_parallel.o naivedh_epi8.o naivesh_epi8.o fcSieve.o bgj1.o bgjf32.o dual.o
OBJ += vec.o quad.o utils.o lattice.o lll.o UidHashTable.o pool.o 3Sieve.o svp.o bkz.o sampler.o lsfdh_epi8.o lsfsh_epi8.o
OBJ += bgj_amx.o bucket_amx.o booster_amx.o
CXX = clang++
NVCC ?= nvcc
CFLAGS = -O3 -g -fopenmp=libomp -march=native -pthread -stdlib=libc++ -Wno-deprecated-declarations
LIBS = -L../dep/ntl/lib -lntl -L../dep/gmp/lib -lgmp -lm
LIBDIR = 
INC_DIR = -I../include/ -I../dep/ntl/include
UID_BACKEND ?= gtl
CXXSTD ?=
ON_CYGWIN = false
CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda
CUDA_CXX ?= g++
CUDA_SMS ?= 80

ifeq ($(ON_CYGWIN), true)
	CFLAGS += -D__ON_CYGWIN
endif

ifeq ($(UID_BACKEND),gtl)
	ifeq ($(strip $(CXXSTD)),)
		CXXSTD = -std=c++17
	endif
	CFLAGS += -DUSE_GTL_UID_TABLE=1 -DUSE_SPARSEPP=0
	INC_DIR += -I../dep/gtl/include
endif

CFLAGS += $(CXXSTD)

ifeq ($(CUDA), 1)
	OBJ += cuda_kernels.o
	CFLAGS += -DHAVE_CUDA
	CUDA_FLAGS += -O3 -std=c++11 -I$(CUDA_PATH)/include
	CUDA_NVCC_FLAGS += -ccbin $(CUDA_CXX) $(CUDA_FLAGS)
	CUDA_NVCC_FLAGS += $(foreach sm,$(CUDA_SMS),-gencode arch=compute_$(sm),code=sm_$(sm))
	CUDA_NVCC_FLAGS += -gencode arch=compute_$(lastword $(CUDA_SMS)),code=compute_$(lastword $(CUDA_SMS))
endif

all: libllib.a

libllib.a: $(OBJ)
	rm -f libllib.a
	ar rcs libllib.a $(OBJ)

%.o: %.cpp 
	$(CXX) $(CFLAGS) $^ $(INC_DIR) $(LIBDIR) -c

%.o: %.cu
	$(NVCC) $(CUDA_NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) cuda_kernels.o libllib.a
