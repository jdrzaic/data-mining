NVCCFLAGS=-std=c++11 -O3

all: build/cg

build/cg: src/cg.cpp include/matrix.h
	nvcc -o build/cg -Iinclude src/cg.cpp $(NVCCFLAGS)

clean:
	rm -rf build/*

