HOST_COMPILER  = g++
INCLUDES  := -I/home/kaller/cuda-samples/Common
CUDA_PATH     ?= /usr/local/cuda
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
# NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
# SM 			   =60 # for gtx 1060
SM 			   =60
GENCODE_FLAGS  = -gencode arch=compute_$(SM),code=sm_$(SM)

SRCS = main.cu
INCS = pixel.h vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h world.h

main: main.o
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main main.o

main.o: $(SRCS) $(INCS)
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

ppm: main
	rm -f out.ppm
	./main > out.ppm

run: main
	./main

profile: main
	rm -f out.ppm
	nvprof ./main > out.ppm

# NOT SUPPORTED ON WSL2
profile_metrics: main
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./main > out.ppm

clean:
	rm -f main main.o
	rm -rf ./bin/
	rm -f out.ppm