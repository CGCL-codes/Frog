
exp : main.cu graph.o partition.o bfs.o cc.o sssp.o pr.o
	nvcc -arch=sm_20 -o exp main.cu obj/graph.o obj/partition.o obj/bfs.o obj/cc.o obj/sssp.o obj/pr.o

graph.o partition.o bfs.o cc.o sssp.o pr.o: | obj

obj:
	mkdir -p obj

graph.o : src/graph.c
	gcc -c -o obj/graph.o src/graph.c

partition.o : src/partition.c
	gcc -c -o obj/partition.o src/partition.c

bfs.o : bfs.cu
	nvcc -c -arch=sm_20 -o obj/bfs.o bfs.cu

cc.o : cc.cu
	nvcc -c -arch=sm_20 -o obj/cc.o cc.cu

sssp.o : sssp.cu
	nvcc -c -arch=sm_20 -o obj/sssp.o sssp.cu

pr.o : pr.cu
	nvcc -c -arch=sm_20 -o obj/pr.o pr.cu
