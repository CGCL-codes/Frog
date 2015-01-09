NVFLAGS=-arch=sm_20
NVCC=/usr/local/cuda/bin/nvcc

tests: tests/bfs_small tests/pagerank_small tests/component_small

tests/%: tests/%.cu
		@mkdir -p bin/$(@D)
			$(NVCC) $(NVFLAGS) $@.cu -o bin/$@
