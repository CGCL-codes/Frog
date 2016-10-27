#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#include "src/frog.cuh"

void bfs_experiments(const Graph * const g);
void cc_experiments(const Graph * const g);
void sssp_experiments(const Graph * const g);
void pr_experiments(const Graph * const g);

#define Experiments(g) \
	bfs_experiments(g); \
cc_experiments(g); \
sssp_experiments(g); \
pr_experiments(g)

int main(int argc, char ** argv) {

	if (argc < 2) {
		printf("Need Input File Name!\n");
		return 0;
	}

	// Choose GPU Device
	CudaSetDevice(0);
	// test Files
	for (int i = 1; i < argc; i++) {
		printf("Reading File ... ");
		timer_start();
		Graph * g = read_graph_bin(argv[i]);
		if (g == NULL) {
			perror("Failed !");
			exit(1);
		} else {
			printf("%.2f ms \n", timer_stop());
			printf("Begin Experiments on Graph (V=%d E=%d File='%s')\n",
					g->vertex_num, g->edge_num, argv[i]);
           
			Experiments(g);

			printf("-------------------------------------------------------------------\n");
			printf("Done Experiments\n");
			release_graph(g);
		}
	}

	return 0;
}
