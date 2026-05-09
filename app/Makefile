BIN = bgj_epi8 svp_solver pump_red svp_tool svp_check lattice_preprocess
CXX = clang++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp=libomp -march=native -stdlib=libc++ -pthread -Wno-deprecated-declarations
LIBS = -L../src -lllib -L../dep/ntl/lib -lntl -L../dep/gmp/lib -lgmp -lm
LIBDIR =
INC_DIR = -I../include/ -I../dep/ntl/include
UID_BACKEND ?= gtl
CXXSTD ?=
CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda
FPLLL_PREFIX ?= ../dep/fplll/install
FPLLL ?= $(if $(wildcard $(FPLLL_PREFIX)/include/fplll/fplll.h),1,0)

ON_CYGWIN = false

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
	CFLAGS += -DHAVE_CUDA
	CUDA_LIBDIR = $(if $(filter x86_64 ppc64le powerpc64le,$(shell uname -m)),lib64,lib)
	LIBS += -L$(CUDA_PATH)/$(CUDA_LIBDIR) -Wl,-rpath=$(CUDA_PATH)/$(CUDA_LIBDIR) -lcudart -lcublas
endif

ifeq ($(FPLLL), 1)
	CFLAGS += -DHAVE_FPLLL
	INC_DIR += -I$(FPLLL_PREFIX)/include
	LIBS += -L$(FPLLL_PREFIX)/lib -Wl,-rpath=$(abspath $(FPLLL_PREFIX)/lib) -Wl,-rpath=$(abspath ../dep/gmp/lib) -lfplll -lmpfr
endif

all: $(BIN)

clean:
	rm -f $(BIN)

%: bin_%.cpp ../src/libllib.a
	$(CXX) $(CFLAGS) $^ $(INC_DIR) $(LIBDIR) $(LIBS) -o $*
