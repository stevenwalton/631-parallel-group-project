CC =  gcc
CPP = g++
NVCC = /usr/local/cuda-10.1/bin/nvcc
CCFLAGS  = -O2 -fopenmp -lm -std=c++11
CFLAGS += -O2 -fopenmp -lm -std=c11
#/usr/local/cuda-10.1/samples/common/inc/helper_cuda.h

CUDA_INSTALL_PATH ?= /usr/local/cuda-10.1/
TOOLKIT_INSTALL_PATH ?= /usr/local/cuda-10.1/samples
CUDA_LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcublas -lcurand -lcusolver 
NVCCFLAGS = -gencode arch=compute_75,code=sm_75 #--ptxas-options=-v
NVCCINC += -I$(CUDA_INSTALL_PATH)/include 
NVCCINC += -I$(TOOLKIT_INSTALL_PATH)/common/inc

SRC  = $(notdir $(wildcard *.cpp)) $(notdir $(wildcard *.cu)) $(notdir $(wildcard *.c))

OBJS = $(addsuffix .o, $(basename $(SRC)))
EXEC = main

all: ${EXEC}

${EXEC}: ${OBJS}
	${CPP} ${CFLAGS} ${LDFLAGS} $^ -o $@ ${LIBS} ${CUDA_LDFLAGS}

%.o : %.c
	$(CC) ${CFLAGS} ${INC} -c $< -o $@ ${LIBS} ${CUDA_LD_FLAGS} ${NVCCINC}

%.o : %.cc
	$(CPP) ${CCFLAGS} ${INC} -c $< -o $@ ${LIBS} ${CUDA_LD_FLAGS} ${NVCCINC}

%.o : %.cu
	$(NVCC) ${NVCCCFLAGS} ${INC} -c $< -o $@ ${CUDA_LD_FLAGS} ${NVCCFLAGS} ${NVCCINC}

clean:
	rm -f *.o *.d *~ *.a *.so *.s ${EXEC}

