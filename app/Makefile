BIN = bgj_epi8 svp_solver pump_red svp_tool svp_check lattice_preprocess
CXX = clang++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp=libomp -march=native -stdlib=libc++ -pthread -Wno-deprecated-declarations
LIBS = -L../src -lllib -L../dep/ntl/lib -lntl -L../dep/gmp/lib -lgmp -lm
LIBDIR =
INC_DIR = -I../include/ -I../dep/ntl/include
CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda

ON_CYGWIN = false

ifeq ($(ON_CYGWIN), true)
	CFLAGS += -D__ON_CYGWIN
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DHAVE_CUDA
	CUDA_LIBDIR = $(if $(filter x86_64 ppc64le powerpc64le,$(shell uname -m)),lib64,lib)
	LIBS += -L$(CUDA_PATH)/$(CUDA_LIBDIR) -Wl,-rpath=$(CUDA_PATH)/$(CUDA_LIBDIR) -lcudart -lcublas
endif

all: $(BIN)

clean:
	rm -f $(BIN)

%: bin_%.cpp ../src/libllib.a
	$(CXX) $(CFLAGS) $^ $(INC_DIR) $(LIBDIR) $(LIBS) -o $*
