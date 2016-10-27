#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"

#define PAR_NUM 20
#define SOURCE_VERTEX 3

#define SSSP_INFINITE -1
#define SSSP_NO_PATH -1

// check if arrays v1 & v2 have the same first n elements (no boundary check)
static void check_values(const int * const v1, const int * const v2, int n) {
	for (int i = 0; i < n; i++) {
		if (v1[i] != v2[i]) {
			printf("Check Fail\n");
			return;
		}
	}
	printf("Check PASS\n");
}

// Dijstra algorithm is too slow, use Bellman Ford algorithm
static void sssp_on_cpu(
		const int vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		const int * const edge_weight,
		const int source,
		int * const dist,
		int * const path
		) {
	// Initializing values
	memset(dist, SSSP_INFINITE, vertex_num * sizeof(int));
	memset(path, SSSP_INFINITE, vertex_num * sizeof(int));
	dist[source] = 0;
	// Calculating
	timer_start();
	int step;
	for (step = 0; step < vertex_num; step++) {
		int flag = 0;
		for (int v = 0; v < vertex_num; v++) {
			int d = dist[v];
			if (d != SSSP_INFINITE) {
				for (int e = vertex_begin[v]; e < vertex_begin[v + 1]; e++) {
					int dest = edge_dest[e];
					int dd = dist[dest];
					if (dd == SSSP_INFINITE || d + edge_weight[e] < dd) {
						flag = 1;
						dist[dest] = d + edge_weight[e];
						path[dest] = v;
					}
				}
			}
		}
		if (flag == 0) break;
	}
	printf("\t%.2f\tsssp_on_cpu\tstep=%d\n", timer_stop(), step);
}

// SSSP kernel run on edges with inner loop
static __global__ void kernel_edge_loop (
		const int edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		const int * const edge_weight,
		int * const dist,
		int * const path,
		int * const continue_flag
		) {
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	// proceeding loop
	for (int i = index; i < edge_num; i += n) {
		int src = edge_src[i];
		int dest = edge_dest[i];
		if (dist[src] != SSSP_INFINITE &&
				(dist[dest] == SSSP_INFINITE ||
				 dist[dest] > dist[src] + edge_weight[i])) {
			flag = 1;
			dist[dest] = dist[src] + edge_weight[i];
			path[dest] = src;
		}
	}
	if (flag == 1) *continue_flag = 1;
}

// SSSP algorithm run on edge with inner loop
static void gpu_sssp_edge_loop(
		const Graph * const g,
		const int * const edge_weight,
		const int source,
		int * const dist,
		int * const path
		) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// Initializing values
	memset(dist, SSSP_INFINITE, vertex_num * sizeof(int));
	memset(path, SSSP_INFINITE, vertex_num * sizeof(int));
	dist[source] = 0;
	// GPU buffer
	CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_edge_weight, edge_num, edge_weight);
	CudaBufferCopy(int, dev_dist, vertex_num, dist);
	CudaBufferCopy(int, dev_path, vertex_num, path);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = 204;
	int tn = 128;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch kernel
		CudaTimerBegin();
		kernel_edge_loop<<<bn, tn>>>(
				edge_num,
				dev_edge_src,
				dev_edge_dest,
				dev_edge_weight,
				dev_dist,
				dev_path,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(dist, dev_dist, vertex_num * sizeof(int));
	CudaMemcpyD2H(path, dev_path, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tsssp_edge_loop\tstep=%d\t\n",
			execTime, timer_stop(), step - 1);
}

// SSSP algorithm run on edge with inner loop, graph partitioned
static void gpu_sssp_edge_part_loop(
		const Graph * const * const g,
		const struct part_table * const t,
		const int * const * const edge_weight,
		const int source,
		int * const dist,
		int * const path,
		int buffersize
		) {
	Auto_Utility();
	timer_start();
	int part_num = t->part_num;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	// Allocate GPU buffer
//#define BUFFSIZE 1024 * 1024
	// Initializing values
	memset(dist, SSSP_INFINITE, t->vertex_num * sizeof(int));
	memset(path, SSSP_INFINITE, t->vertex_num * sizeof(int));
	dist[source] = 0;
	// GPU buffer
	CudaBufferCopy(int, dev_dist, t->vertex_num, dist);
	CudaBufferCopy(int, dev_path, t->vertex_num, path);
	CudaBufferZero(int, dev_continue_flag, 1);
	/*
	   int ** dev_edge_src = (int **) Calloc(part_num, sizeof(int *));
	   int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
	   int ** dev_edge_weight = (int **) Calloc(part_num, sizeof(int *));
	   for (int i = 0; i < part_num; i++) {
	   int size = g[i]->edge_num * sizeof(int);
	   CudaBufferFill(dev_edge_src[i], size, g[i]->edge_src);
	   CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
	   CudaBufferFill(dev_edge_weight[i], size, edge_weight[i]);
	   }*/
	CudaBuffer(int, dev_edge_src, buffersize);
	CudaBuffer(int, dev_edge_dest, buffersize);
	CudaBuffer(int, dev_edge_weight, buffersize);
	// settings
	int bn = 204;
	int tn = 128;
	int flag = 0;
	int step = 0;
	float ssspTime = 0.0;
	float transTime= 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch kernel
		for (int i = 0; i < part_num; i++) {
			int partSize = g[i]->edge_num;
			for (int offset = 0; offset < partSize; offset += buffersize)
			{
				int curSize = (partSize - offset) > buffersize ? buffersize : (partSize - offset);
				CudaTimerBegin();
				cudaMemcpyAsync(dev_edge_src, g[i]->edge_src + offset, curSize * sizeof(int),cudaMemcpyHostToDevice,stream);
				cudaMemcpyAsync(dev_edge_dest, g[i]->edge_dest + offset, curSize * sizeof(int),cudaMemcpyHostToDevice,stream);
				cudaMemcpyAsync(dev_edge_weight, edge_weight[i] + offset, curSize * sizeof(int),cudaMemcpyHostToDevice,stream);
				transTime += CudaTimerEnd();
				// kernel launch
				CudaTimerBegin();
				kernel_edge_loop<<<bn, tn,0,stream>>>(
						curSize,
						dev_edge_src,
						dev_edge_dest,
						dev_edge_weight,
						dev_dist,
						dev_path,
						dev_continue_flag
						);
				ssspTime += CudaTimerEnd();
			}
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(dist, dev_dist, t->vertex_num * sizeof(int));
	CudaMemcpyD2H(path, dev_path, t->vertex_num * sizeof(int));
//	printf("ssspTime = %.2f ms, transTime = %.2fms\n", ssspTime, transTime);
	printf("\t%.2f\t%.2f\tsssp_edge_part_loop\tstep=%d\t",
			ssspTime, timer_stop(), step - 1);
}

// SSSP kernel run on vertices without inner loop
static __global__ void kernel_vertex (
		const int vertex_num,
		const int * const vertex_begin,
		const int * const edge_dest,
		const int * const edge_weight,
		int * const dist,
		int * const path,
		int * const continue_flag
		) {
	// thread index
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// proceed
	if (i < vertex_num) {
		if (dist[i] != SSSP_INFINITE) {
			int flag = 0;
			for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
				int dest = edge_dest[e];
				if (dist[dest] == SSSP_INFINITE || dist[dest] > dist[i] + edge_weight[e]) {
					flag = 1;
					dist[dest] = dist[i] + edge_weight[e];
					path[dest] = i;
				}
			}
			if (flag == 1) *continue_flag = 1;
		}
	}
}

// SSSP algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_sssp_vertex(
		const Graph * const g,
		const int * const edge_weight,
		const int source,
		int * const dist,
		int * const path
		) {
	Auto_Utility();
	timer_start();
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	// Initializing values
	memset(dist, SSSP_INFINITE, vertex_num * sizeof(int));
	memset(path, SSSP_INFINITE, vertex_num * sizeof(int));
	dist[source] = 0;
	// GPU buffer
	CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
	CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
	CudaBufferCopy(int, dev_edge_weight, edge_num, edge_weight);
	CudaBufferCopy(int, dev_dist, vertex_num, dist);
	CudaBufferCopy(int, dev_path, vertex_num, path);
	CudaBufferZero(int, dev_continue_flag, 1);
	// settings
	int bn = (vertex_num + 255) / 256;
	int tn = 256;
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch kernel
		CudaTimerBegin();
		kernel_vertex<<<bn, tn>>>(
				vertex_num,
				dev_vertex_begin,
				dev_edge_dest,
				dev_edge_weight,
				dev_dist,
				dev_path,
				dev_continue_flag
				);
		execTime += CudaTimerEnd();
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(dist, dev_dist, vertex_num * sizeof(int));
	CudaMemcpyD2H(path, dev_path, vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tsssp_vertex\tstep=%d\t",
			execTime, timer_stop(), step);
}

// SSSP kernel run on vertices without inner loop
static __global__ void kernel_vertex_part (
		const int vertex_num,
		const int * const vertex_id,
		const int * const vertex_begin,
		const int * const edge_dest,
		const int * const edge_weight,
		int * const dist,
		int * const path,
		int * const continue_flag
		) {
	// thread index
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// proceed
	if (index < vertex_num) {
		int i = vertex_id[index];
		if (dist[i] != SSSP_INFINITE) {
			int flag = 0;
			for (int e = vertex_begin[index]; e < vertex_begin[index + 1]; e++) {
				int dest = edge_dest[e];
				if (dist[dest] == SSSP_INFINITE || dist[dest] > dist[i] + edge_weight[e]) {
					flag = 1;
					dist[dest] = dist[i] + edge_weight[e];
					path[dest] = i;
				}
			}
			if (flag != 0) *continue_flag = 1;
		}
	}
}

// BFS algorithm on graph g, partitioned, run on vertices without inner loop
static void gpu_sssp_vertex_part(
		const Graph * const * const g,
		const struct part_table * const t,
		const int * const * const edge_weight,
		const int source,
		int * const dist,
		int * const path
		) {
	Auto_Utility();
	timer_start();
	int part_num = t->part_num;
	// Initializing values
	memset(dist, SSSP_INFINITE, t->vertex_num * sizeof(int));
	memset(path, SSSP_INFINITE, t->vertex_num * sizeof(int));
	dist[source] = 0;
	// GPU buffer
	CudaBufferCopy(int, dev_dist, t->vertex_num, dist);
	CudaBufferCopy(int, dev_path, t->vertex_num, path);
	CudaBufferZero(int, dev_continue_flag, 1);
	int ** dev_vertex_id = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_vertex_begin = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
	int ** dev_edge_weight = (int **) Calloc(part_num, sizeof(int *));
	for (int i = 0; i < part_num; i++) {
		int size = (g[i]->vertex_num + 1) * sizeof(int);
		CudaBufferFill(dev_vertex_begin[i], size, g[i]->vertex_begin);
		size = g[i]->vertex_num * sizeof(int);
		CudaBufferFill(dev_vertex_id[i], size, t->part_vertex[i]);
		size = g[i]->edge_num * sizeof(int);
		CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
		CudaBufferFill(dev_edge_weight[i], size, edge_weight[i]);
	}
	// settings
	int flag = 0;
	int step = 0;
	float execTime = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		// Launch Kernel for this Iteration
		for (int i = 0; i < part_num; i++) {
			CudaTimerBegin();
			kernel_vertex_part<<<(g[i]->vertex_num + 255) / 256, 256>>>
				(
				 g[i]->vertex_num,
				 dev_vertex_id[i],
				 dev_vertex_begin[i],
				 dev_edge_dest[i],
				 dev_edge_weight[i],
				 dev_dist,
				 dev_path,
				 dev_continue_flag
				);
			execTime += CudaTimerEnd();
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
	} while(flag != 0 && step < 100);
	// Copy Back Values
	CudaMemcpyD2H(dist, dev_dist, t->vertex_num * sizeof(int));
	CudaMemcpyD2H(path, dev_path, t->vertex_num * sizeof(int));
	printf("\t%.2f\t%.2f\tsssp_vertex_part\tstep=%d\t",
			execTime, timer_stop(), step - 1);
}

void sssp_experiments(const Graph * const g, int size) {

	printf("-------------------------------------------------------------------\n");
	// partition on the Graph
	printf("Partitioning ... ");
	timer_start();
	struct part_table * t =
		partition(g->vertex_num, g->edge_num, g->vertex_begin, g->edge_dest, PAR_NUM);
	if (t == NULL) {
		perror("Failed !");
		exit(1);
	} else {
		printf("%.2f ms ... ", timer_stop());
	}
	// get Partitions
	printf("Get partitions ... ");
	timer_start();
	Graph ** part = get_cut_graphs(g, t);
	if (part == NULL) {
		perror("Failed !");
		exit(1);
	} else {
		printf("%.2f ms\n", timer_stop());
	}

	int * value_cpu = (int *) calloc(g->vertex_num, sizeof(int));
	int * value_gpu = (int *) calloc(g->vertex_num, sizeof(int));
	int * path = (int *) calloc(g->vertex_num, sizeof(int));
	int * edge_weight = (int *) malloc(g->edge_num * sizeof(int));
	if (value_cpu == NULL || value_gpu == NULL || path == NULL || edge_weight == NULL) {
		perror("Out of Memory for values");
		exit(1);
	}
	for (int i = 0; i < g->edge_num; i++) edge_weight[i] = 1; // no input, set to 1
	int ** part_weight=(int **)calloc(PAR_NUM,sizeof(int *));
	for (int i = 0; i < PAR_NUM; i++) {
		part_weight[i] = (int *) calloc(t->part_edge_num[i], sizeof(int));
		int n = 0;
		for (int k = 0; k < t->part_vertex_num[i]; k++) {
			int v = t->part_vertex[i][k];
			for (int e = g->vertex_begin[v]; e < g->vertex_begin[v + 1]; e++) {
				part_weight[i][n++] = edge_weight[e];
			}
		}
	}

	printf("\tTime\tTotal\tTips\n");
	//sssp_on_cpu(g->vertex_num, g->vertex_begin, g->edge_dest, edge_weight,SOURCE_VERTEX, value_cpu, path);
	/*
	   gpu_sssp_edge_loop(g, edge_weight, SOURCE_VERTEX, value_gpu, path);
	   check_values(value_cpu, value_gpu, g->vertex_num);
	 */
	gpu_sssp_edge_part_loop(part, t, part_weight, SOURCE_VERTEX, value_gpu, path,size);
	//check_values(value_cpu, value_gpu, g->vertex_num);
	/*
	   gpu_sssp_vertex(g, edge_weight, SOURCE_VERTEX, value_gpu, path);
	   check_values(value_cpu, value_gpu, g->vertex_num);
	   gpu_sssp_vertex_part(part, t, part_weight, SOURCE_VERTEX, value_gpu, path);
	   check_values(value_cpu, value_gpu, g->vertex_num);
	 */
	release_table(t);
	for (int i = 0; i < PAR_NUM; i++) release_graph(part[i]);
	free(part);
	free(value_cpu);
	free(value_gpu);
	free(path);
	free(edge_weight);
	for (int i = 0; i < PAR_NUM; i++) free(part_weight[i]);
}
