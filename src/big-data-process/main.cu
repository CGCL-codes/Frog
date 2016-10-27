#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#include "src/frog.cuh"


void bfs_experiments(const Graph * const g, int buffersize);
void cc_experiments(const Graph * const g, int buffersize);
void sssp_experiments(const Graph * const g, int buffersize);
void pr_experiments(const Graph * const g, int buffersize);

#define Experiments(g,buffersize) \
            bfs_experiments(g,buffersize); \
            pr_experiments(g, buffersize); \
            cc_experiments(g, buffersize);\
            sssp_experiments(g, buffersize)

int main(int argc, char ** argv) {

    if (argc < 3) {
        printf("Need Input File Name!\n");
        return 0;
    }

    // Choose GPU Device
    CudaSetDevice(3);
    int size=atoi(argv[2]);
    // test Files
    for (int i = 1; i < 2; i++) {
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

            Experiments(g,size);

            printf("\nDone Experiments\n");
            release_graph(g);
        }
    }

    return 0;
}
