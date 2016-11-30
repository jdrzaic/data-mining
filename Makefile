NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 $(DEFINE)

INCLUDE= \
	include/constants.h	\
	include/error.h		\
	include/lapack.h	\
	include/matrix.h 	\
	include/page_rank.h	\

LIBS= \
	-lcublas	\
	-lcusparse	\
	-lblas		\
	-llapack	\


all: build/cg build/pr

build/cg: src/cg.cpp $(INCLUDE)
	$(NVCC) -o $@ -Iinclude $< $(NVCCFLAGS)

build/pr: src/main.cpp src/page_rank.cpp $(INCLUDE)
	$(NVCC) -o $@ -Iinclude src/main.cpp src/page_rank.cpp $(NVCCFLAGS) \
		$(LIBS)

clean:
	rm -rf build/*

