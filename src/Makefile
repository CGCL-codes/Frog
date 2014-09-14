NVFLAGS=-arch=sm_20
NVCC=/usr/local/cuda/bin/nvcc

tests: tests/pagerank tests/bfs

tests/%: tests/%.cu
		@mkdir -p bin/$(@D)
			$(NVCC) $(NVFLAGS) $@.cu -o bin/$@
