#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"

// print info about CC values
void report_cc_values(const int * const values, int n) {
    int * c = (int *)calloc(n, sizeof(int));
    int cc = 0;
    for (int i = 0; i < n; i++) {
        int r = values[i];
        if (c[r] == 0) cc++;
        c[r]++;
    }
    printf("Number of Connected Components: %d\n", cc);
    int k = 0;
    printf("\tID\tRoot\tN\n");
    for (int i = 0; i < n; i++) {
        if (c[i] != 0)
            printf("\t%d\t%d\t%d\n", k++, i, c[i]);
        if (k > 20) {
            printf("\t...\n");
            break;
        }
    }
    free(c);
}

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

// MFset function - Reset
static void Reset(int s[], int n) {
    for (int i = 0; i < n; i++) s[i] = i;
}
// MFset function - Find
static int Find(int s[], int x) {
    int root = x;
    // find the root
    while (s[root] != root)
        root = s[root];
    // merge the path
    while (s[x] != root) {
        int t = s[x];
        s[x] = root;
        x = t;
    }
    // return root
    return root;
}
// MFset function - Union
static void Union(int s[], int x, int y) {
    int root1 = Find(s, x);
    int root2 = Find(s, y);
    // keep the smaller root
    if (root1 < root2)
        s[root2] = root1;
    else if (root1 > root2)
        s[root1] = root2;
}
// CC on CPU
static void cc_on_cpu(
    const int vertex_num,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values
) {
    timer_start();
    // Initializing values
    Reset(values, vertex_num);
    // MFset calculating
    for (int i = 0; i < vertex_num; i++)
        for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++)
            Union(values, i, edge_dest[k]);
    // Calculating final values
    for (int i = 0; i < vertex_num; i++)
        Find(values, i);
    printf("\t%.2f\tCC on CPU\n", timer_stop());
}

// CC Kernel on edges with inner loops
static __global__ void kernel_edge_loop (
    int const edge_num,
    const int * const edge_src,
    const int * const edge_dest,
    int * const values,
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
        if (values[src] != values[dest]) {
            flag = 1;
            // combined to the smaller Component ID
            if (values[src] > values[dest])
                values[src] = values[dest];
            else
                values[dest] = values[src];
        }
    }
    if (flag == 1) *continue_flag = 1;
}

// CC algorithm on graph g, not partitioned, run on edges with inner loop
static void gpu_cc_edge_loop(const Graph * const g, int * const values) {
    Auto_Utility();
    timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    // GPU buffer
    for (int i = 0; i < vertex_num; i++) values[i] = i;
    CudaBufferCopy(int, dev_edge_src, edge_num, g->edge_src);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferCopy(int, dev_value, vertex_num, values);
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
        CudaTimerBegin();
        kernel_edge_loop<<<bn, tn>>>(
            edge_num,
            dev_edge_src,
            dev_edge_dest,
            dev_value,
            dev_continue_flag
        );
        execTime += CudaTimerEnd();
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag != 0 && step < 100);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tcc_edge_loop\tstep=%d\t",
            execTime, timer_stop(), step);
}

// CC algorithm on graph g, partitioned, run on edges with inner loop
static void gpu_cc_edge_part_loop(
    const Graph * const * const g,
    const struct part_table * const t,
    int * const values
) {
    Auto_Utility();
    timer_start();
    int part_num = t->part_num;
    // GPU buffer
    for (int i = 0; i < t->vertex_num; i++) values[i] = i;
    int ** dev_edge_src = (int **) Calloc(part_num, sizeof(int *));
    int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
    for (int i = 0; i < part_num; i++) {
        int size = g[i]->edge_num * sizeof(int);
        CudaBufferFill(dev_edge_src[i], size, g[i]->edge_src);
        CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
    }
    CudaBufferCopy(int, dev_value, t->vertex_num, values);
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
        // Launch Kernel for this Iteration
        for (int i = 0; i < part_num; i++) {
            CudaTimerBegin();
            kernel_edge_loop<<<bn, tn>>>
            (
                g[i]->edge_num,
                dev_edge_src[i],
                dev_edge_dest[i],
                dev_value,
                dev_continue_flag
            );
            execTime += CudaTimerEnd();
        }
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag != 0 && step < 100);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, t->vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tpart_cc_edge_loop\tstep=%d\t",
            execTime, timer_stop(), step);
}

// CC Kernel on vertices without inner loops
static __global__ void kernel_vertex (
    int const vertex_num,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values,
    int * const continue_flag
) {
    // thread index of this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // proceed
    if (i < vertex_num) {
        int new_value = values[i];
        int flag = 0;
        // find the best new_value (smallest)
        for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
            int dest_value = values[edge_dest[e]];
            if (dest_value != new_value)
                flag = 1;
            if (dest_value < new_value)
                new_value = dest_value;
        }
        // update values
        if (flag) {
            values[i] = new_value;
            for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
                values[edge_dest[e]] = new_value;
            }
            *continue_flag = 1;
        }
    }
}

// CC algorithm on graph g, not partitioned, run on vertices without inner loop
static void gpu_cc_vertex(const Graph * const g, int * const values) {
    Auto_Utility();
    timer_start();
    int vertex_num = g->vertex_num;
    int edge_num = g->edge_num;
    // GPU buffer
    for (int i = 0; i < vertex_num; i++) values[i] = i;
    CudaBufferCopy(int, dev_vertex_begin, vertex_num, g->vertex_begin);
    CudaBufferCopy(int, dev_edge_dest, edge_num, g->edge_dest);
    CudaBufferCopy(int, dev_value, vertex_num, values);
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
        CudaTimerBegin();
        kernel_vertex<<<bn, tn>>>(
            vertex_num,
            dev_vertex_begin,
            dev_edge_dest,
            dev_value,
            dev_continue_flag
        );
        execTime += CudaTimerEnd();
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag != 0 && step < 100);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tcc_vertex\tstep=%d\t",
            execTime, timer_stop(), step);
}

// CC Kernel on vertices without inner loops
static __global__ void kernel_vertex_part (
    int const vertex_num,
    const int * const vertex_id,
    const int * const vertex_begin,
    const int * const edge_dest,
    int * const values,
    int * const continue_flag
) {
    // thread index of this thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // proceed
    if (i < vertex_num) {
        int id = vertex_id[i];
        int new_value = values[id];
        int flag = 0;
        // find the best new_value (smallest)
        for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++) {
            int dest_value = values[edge_dest[e]];
            if (dest_value < new_value) {
                flag = 1;
                new_value = dest_value;
            }
        }
        // update values
        if (flag) {
            values[id] = new_value;
            *continue_flag = 1;
        }
    }
}

// BFS algorithm on graph g, partitioned, run on vertices without inner loop
static void gpu_cc_vertex_part(
    const Graph * const * const g,
    const struct part_table * const t,
    int * const values
) {
    Auto_Utility();
    timer_start();
    int part_num = t->part_num;
    // GPU buffer
    int ** dev_vertex_begin = (int **) Calloc(part_num, sizeof(int *));
    int ** dev_edge_dest = (int **) Calloc(part_num, sizeof(int *));
    int ** dev_vertex_id = (int **) Calloc(part_num, sizeof(int *));
    for (int i = 0; i < part_num; i++) {
        int size = (g[i]->vertex_num + 1) * sizeof(int);
        CudaBufferFill(dev_vertex_begin[i], size, g[i]->vertex_begin);
        size = g[i]->edge_num * sizeof(int);
        CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
        size = g[i]->vertex_num * sizeof(int);
        CudaBufferFill(dev_vertex_id[i], size, t->part_vertex[i]);
    }
    for (int i = 0; i < t->vertex_num; i++) values[i] = i;
    CudaBufferCopy(int, dev_value, t->vertex_num, values);
    CudaBufferZero(int, dev_continue_flag, 1);
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
                dev_value,
                dev_continue_flag
            );
            execTime += CudaTimerEnd();
        }
        // Copy Back Flag
        CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
        step++;
    } while(flag != 0 && step < 100);
    // Copy Back Values
    CudaMemcpyD2H(values, dev_value, t->vertex_num * sizeof(int));
    printf("\t%.2f\t%.2f\tpart_cc_vertex\tstep=%d\t",
            execTime, timer_stop(), step);
}

// experiments of BFS on Graph g with Partition Table t and partitions
void cc_experiments(const Graph * const g) {

    printf("-------------------------------------------------------------------\n");
    // partition on the Graph
    printf("Partitioning ... ");
    timer_start();
    struct part_table * t =
        partition(g->vertex_num, g->edge_num, g->vertex_begin, g->edge_dest, 5);
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
    if (value_cpu == NULL || value_gpu == NULL) {
        perror("Out of Memory for values");
        exit(1);
    }

    printf("\tTime\tTotal\tTips\n");

    cc_on_cpu(g->vertex_num, g->vertex_begin, g->edge_dest, value_cpu);
    //report_cc_values(value_cpu, g->vertex_num);

    //gpu_cc_edge_loop(g, value_gpu);
    //check_values(value_cpu, value_gpu, g->vertex_num);
    gpu_cc_edge_part_loop(part, t, value_gpu);
    check_values(value_cpu, value_gpu, g->vertex_num);

	/*
    gpu_cc_vertex(g, value_gpu);
    check_values(value_cpu, value_gpu, g->vertex_num);
    gpu_cc_vertex_part(part, t, value_gpu);
    check_values(value_cpu, value_gpu, g->vertex_num);
  */
    release_table(t);
    for (int i = 0; i < 5; i++) release_graph(part[i]);
    free(part);
    free(value_cpu);
    free(value_gpu);
}

