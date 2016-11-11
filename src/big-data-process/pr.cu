#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/frog.cuh"

#define PAR_NUM 20
#define PAGERANK_COEFFICIENT 0.85f
#define PAGERANK_INITIAL 1.0f
#define PAGERANK_THRESHOLD 0.005f

// check if arrays v1 & v2 have the same first n elements (no boundary check)
static void check_values(const float * const v1, const float * const v2, int n) {
	for (int i = 0; i < n; i++) {
		if (fabs(v1[i] - v2[i]) > PAGERANK_THRESHOLD) {
			printf("First difference at %d : %f & %f\n", i, v1[i], v2[i]);
			return;
		}
	}
	printf("Check PASS\n\n");
}

static void get_degree(
		const Graph * const m,
		int * const in_degree,
		int * const out_degree
		)
{
	int vertex_num=m->vertex_num;
	int *vertex_begin=m->vertex_begin;
	int *edge_dest=m->edge_dest;
	for (int i = 0; i < vertex_num; ++i)
	{
		out_degree[i]=vertex_begin[i+1]-vertex_begin[i];
		for (int j = vertex_begin[i]; j < vertex_begin[i+1]; ++j)
		{
			in_degree[edge_dest[j]]++;
		}
	}
}

/* 
 * kernel function
 */
// Iterative Kernel of PageRank Algorithm : checking values
__global__ void kernel_PageRank_check (
		int const vertex_num,
		const int * const vertex_degree,
		float * const values,
		float * const tem_values,
		float * const add_values,
		int * const continue_flag
		) 
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int flag = 0;
	// proceeding loop
	for (int i = index; i < vertex_num; i += n) {
		float new_value = PAGERANK_COEFFICIENT * tem_values[i] + 1.0f - PAGERANK_COEFFICIENT;
		if (fabs(new_value - values[i]) > PAGERANK_THRESHOLD)
			flag = 1;
		if (vertex_degree[i] > 0)
			add_values[i] = new_value / vertex_degree[i];
		values[i] = new_value;
		tem_values[i] = 0.0f;
	}
	if (flag == 1) *continue_flag = 1;
}

// Iterative Kernel of PageRank Algorithm : adding values
__global__ static void kernel_calc (
		int const edge_num,
		const int * const edge_src,
		const int * const edge_dest,
		const float * const add_values,
		float * const tem_values
		) 
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// per_thread elements for each thread, left_out elements threads in front
	int per_thread = edge_num / n;
	int left_out = edge_num - per_thread * n;
	// calculate the start & end element for this thread
	int element_start = per_thread * index +
		(index > left_out ? left_out : index);
	int element_end = per_thread * (index + 1) +
		((index + 1) > left_out ? left_out : (index  + 1));
	// proceeding loop
	float sum = 0.0f;
	int prev = edge_dest[element_start];
	for (int i = element_start; i < element_end; i++) {
		if (prev == edge_dest[i]) {
			// Accumulate values for that dest vertex
			sum += add_values[edge_src[i]];
		} else {
			// The thread scans a continuous block of edges sorted by edge_dest
			// So, tem_values are updated only when arrives a new dest vertex
			atomicAdd(&tem_values[prev], sum);
			sum = add_values[edge_src[i]];
			prev = edge_dest[i];
		}
	}
	// The last one
	if (sum > 0) atomicAdd(&tem_values[prev], sum);
}

// Iterative Kernel of PageRank Algorithm : adding values
__global__ static void kernel_vertex_pr (
		int const vertex_num,
		const int * const vertex_id,
		const int * const vertex_begin,
		const int * const edge_src,
		float * const values,
		float * const add_values,
		int * const tem_values,
		int * const continue_flag
		) 
{
	// total thread number & thread index of this thread
	int n = blockDim.x * gridDim.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// continue flag for each thread
	int  flag = 0;
	// proceeding loop
	for (int i = index; i < vertex_num; i += n) {
		int id=vertex_id[i];
		if(tem_values[id]==1)
			continue;
		float new_value = 0.0f;
		int nbrs_num=vertex_begin[i+1]-vertex_begin[i];
		for (int e = vertex_begin[i]; e < vertex_begin[i + 1]; e++)
			new_value += add_values[edge_src[e]];
		new_value = PAGERANK_COEFFICIENT * new_value + 1.0f - PAGERANK_COEFFICIENT;
		if (fabs(new_value - values[id]) > PAGERANK_THRESHOLD)
			flag = 1;
		if(fabs(new_value - values[id]) < PAGERANK_THRESHOLD)
			tem_values[id]=1;
		if (nbrs_num > 0)
			add_values[id] = new_value /nbrs_num;
		values[id] = new_value;
	}
	// update flag
	if (flag == 1) *continue_flag = 1;
}

static void gpu_pr_atomic(
		const Graph * const * const g,
		const struct part_table * const t,
		const int * const out_degree,
		float * const values,
		int buffersize
		)
{
	//printf("PageRank with atomic function calls on GPU ... \n");
	//pre-process the graph to get the degree of each vertices
	int vertex_num=t->vertex_num;
	//int edge_num=m->vertex_num;

	//Choose GPU Device
	CudaSetDevice(0);
	int part_num=t->part_num;
	Auto_Utility();
	cudaStream_t stream;
	cudaStreamCreate(&stream);
   //#define BUFFSIZE 1024*1024
	CudaBuffer(int, dev_edge_src, buffersize);
	CudaBuffer(int, dev_edge_dest, buffersize);
	/*
	//Allocate GPU buffer
	int ** dev_edge_src = (int **) calloc(part_num, sizeof(int *));
	int ** dev_edge_dest = (int **) calloc(part_num, sizeof(int *));
	//int *rev_begin=(int *)calloc(vertex_num+1,sizeof(int));
	for (int i = 0; i < part_num ; ++i)
	{
	//preprocess_degree_edge(g[i],in_degree,out_degree,rev_begin,dev_edge_src[i],dev_edge_dest[i]);
	int size=(g[i]->edge_num) * sizeof(int);
	CudaBufferFill(dev_edge_src[i], size, g[i]->edge_src);
	CudaBufferFill(dev_edge_dest[i], size, g[i]->edge_dest);
	}
	 */
	CudaBufferCopy(int, dev_vertex_degree,vertex_num,out_degree);
	CudaBufferZero(float, dev_value, vertex_num);
	CudaBufferZero(float, dev_tem_value, vertex_num);
	CudaBufferZero(float, dev_add_value,vertex_num);
	CudaBufferZero(int, dev_continue_flag, 1);
	int bn = 204;	
	int tn = 128;

	int flag = 0;
	int step = 0;
	float prTime = 0.0;
	float transTime = 0.0;
	float tm = 0.0;

	timer_start();
	CudaTimerBegin();
	kernel_PageRank_check<<<bn, tn>>>(
			vertex_num,
			dev_vertex_degree,
			dev_value,
			dev_tem_value,
			dev_add_value,
			dev_continue_flag
			);
	tm=CudaTimerEnd();
	prTime += tm;
	//printf("\tPageRank Values Initialized, Time = %.2f ms\n", tm);
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(int));
	/*
	   printf("\tvalues:");
	   for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
	   printf(" ...\n");
	 */

	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		for (int i = 0; i < part_num; i++)
		{
			int partSize=g[i]->edge_num;
			for(int offset=0;offset<partSize;offset+=buffersize)
			{	
				int curSize=(partSize-offset)>buffersize?buffersize:(partSize - offset);
				CudaTimerBegin();
				cudaMemcpyAsync(dev_edge_src, g[i]->edge_src + offset, curSize * sizeof(int),cudaMemcpyHostToDevice,stream);
				cudaMemcpyAsync(dev_edge_dest, g[i]->edge_dest + offset, curSize * sizeof(int),cudaMemcpyHostToDevice,stream);
				transTime+=CudaTimerEnd();
				CudaTimerBegin();
				kernel_calc<<<bn, tn,0,stream>>>(
						curSize,
						dev_edge_dest,
						dev_edge_src,
						dev_add_value,
						dev_tem_value
						);
				prTime+= CudaTimerEnd();
			}
		}
		CudaTimerBegin();
		kernel_PageRank_check<<<bn, tn,0, stream>>>(
				vertex_num,
				dev_vertex_degree,
				dev_value,
				dev_tem_value,
				dev_add_value,
				dev_continue_flag
				);
		tm=CudaTimerEnd();
		prTime += tm;
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;
		CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(float));
		/*
		   printf("\tvalues:");
		   for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
		   printf(" ...\n");
		 */

	} while(flag != 0 && step < 10);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(float));
//	printf("prTime = %.2f ms....., transTime = %.2fms\n", prTime, transTime);
	printf("\t%.2f\t%.2f\tpagerank_edge_loop\tstep=%d\t\n",
			prTime,timer_stop(),  step);
	/*
	   printf("\tvalues:");
	   for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
	   printf(" ...\n");
	 */
}
static void gpu_pr_navie(
		const Graph * const * const g, 
		const struct part_table * const t, 
		const int * out_degree,
		float  * const values)
{
	//printf("PageRank on Reverse Edge List without atomic function calls on GPU ... \n");
	int vertex_num=t->vertex_num;
	int part_num=t->part_num;
	Auto_Utility();

	//Allocate GPU buffer
	int ** dev_vertex_begin = (int **) calloc(part_num, sizeof(int *));
	int ** dev_edge_src = (int **) calloc(part_num, sizeof(int *));
	int ** dev_vertex_id =(int **)calloc(part_num,sizeof(int *));

	for(int i=0;i<part_num;i++)
	{
		int size=(g[i]->vertex_num+1)*sizeof(int);
		CudaBufferFill(dev_vertex_begin[i], size,g[i]->vertex_begin);
		size= g[i]->edge_num*sizeof(int);
		CudaBufferFill(dev_edge_src[i], size,g[i]->edge_dest);
		size = g[i]->vertex_num * sizeof(int);
		CudaBufferFill(dev_vertex_id[i], size, t->part_vertex[i]);
	}
	CudaBufferCopy(int,dev_vertex_degree,vertex_num,out_degree);
	CudaBufferZero(float, dev_value, vertex_num);
	CudaBufferZero(int, dev_tem_value, vertex_num);
	CudaBufferZero(float, dev_add_value,vertex_num);
	CudaBufferZero(int, dev_continue_flag, 1);
	//printf("Done,Time=%.2f ms\n",timer_stop());

	int bn = 256;
	int tn = 128;

	int flag = 0;
	int step = 0;
	float totalTime = 0.0;
	float tm = 0.0;
	// Main Loop
	do {
		// Clear Flag
		CudaMemset(dev_continue_flag, 0, sizeof(int));
		for (int i = 0; i < part_num; ++i)
		{
			CudaTimerBegin();
			bn=(g[i]->vertex_num+255)/256;
			tn=256;
			kernel_vertex_pr<<<bn, tn>>>(
					g[i]->vertex_num,
					dev_vertex_id[i],
					dev_vertex_begin[i],
					dev_edge_src[i],
					dev_value,
					dev_add_value,
					dev_tem_value,
					dev_continue_flag
					);
			tm=CudaTimerEnd();
			totalTime += tm;
		}
		// Copy Back Flag
		CudaMemcpyD2H(&flag, dev_continue_flag, sizeof(int));
		step++;

		CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(float));
		/*   
			 printf("\tvalues:");
			 for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
			 printf(" ...\n");
		 */

	} while(flag != 0 && step < 20);
	// Copy Back Values
	CudaMemcpyD2H(values, dev_value, vertex_num * sizeof(float));
	printf("\t%.2f\t%.2f\tpagerank_vertex_loop\tstep=%d\t\n",
			totalTime,timer_stop(),  step);
	/*
	   printf("\tvalues:");
	   for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
	   printf(" ...\n");
	 */
}
static void cpu_pagerank(
		const Graph * const g, 
		float * const values)
{

	//printf("PageRank on CPU ... \n");
	int vertex_num = g->vertex_num;
	int edge_num = g->edge_num;
	Auto_Utility();
	//printf("\tPreparing graph data ... ");
	timer_start();
	int * vertex_begin = g->vertex_begin;
	int * edge_dest = g->edge_dest;
	int * in_degree = (int *) calloc(vertex_num,sizeof(int));
	int * out_degree = (int *) calloc(vertex_num ,sizeof(int));
	int * rev_begin = (int *) calloc((vertex_num + 1) ,sizeof(int));
	int * rev_dest = (int *) calloc(edge_num ,sizeof(int));
	int * rev_src = (int *) calloc(edge_num,sizeof(int));
	// count degrees
	for (int i = 0; i < vertex_num; i++) {
		out_degree[i] = vertex_begin[i + 1] - vertex_begin[i];
		for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++)
			in_degree[edge_dest[k]]++;
	}
	// calculate edge offset numbers
	int tem = 0;
	for (int i = 0; i < vertex_num; i++) {
		rev_begin[i] = tem;
		tem += in_degree[i];
	}
	rev_begin[vertex_num] = tem;
	// reorder edges
	for (int i = 0; i < vertex_num; i++)
		for (int k = vertex_begin[i]; k < vertex_begin[i + 1]; k++) {
			int dest = edge_dest[k];
			int index = rev_begin[dest];
			rev_dest[index] = dest;
			rev_src[index] = i;
			rev_begin[dest] = index + 1;
		}
	// reset edge offset numbers
	tem = 0;
	for (int i = 0; i < vertex_num; i++) {
		rev_begin[i] = tem;
		tem += in_degree[i];
	}
	rev_begin[vertex_num] = tem;
	//printf("Done, Time = %.2f ms\n", timer_stop());

	//printf("\tInitializing values ... ");
	timer_start();
	double * value_add = (double *) calloc(vertex_num,sizeof(double));
	double * value_tem = (double *) calloc(vertex_num,sizeof(double));
	// initail values
	for (int i = 0; i < vertex_num; i++) {
		if (out_degree[i])
			value_add[i] = double(1.0f - PAGERANK_COEFFICIENT) / out_degree[i];
	}
	//printf("Done, Time = %.2f ms\n", timer_stop());

	timer_start();
	int flag = 1;
	int step = 1;
	while (flag != 0 && step < 10) {
		flag = 0;
		//printf("\tIteration: %d\n", step);
		for (int i = 0; i < edge_num; i++) {
			value_tem[rev_dest[i]] += value_add[rev_src[i]];
		}
		for (int i = 0; i < vertex_num; i++) {
			double new_value = PAGERANK_COEFFICIENT * value_tem[i] + 1.0 - PAGERANK_COEFFICIENT;
			if (flag == 0 && fabs(new_value - values[i]) > PAGERANK_THRESHOLD)
				flag = 1;
			if (out_degree[i] > 0)
				value_add[i] = new_value / out_degree[i];
			values[i] = new_value;
			value_tem[i] = 0.0;
		}
		step++;
	}
	printf("\t%.2f\tPageRank on CPU \tStep=%d\n", timer_stop(),step);
	/*
	   printf("\tvalues:");
	   for (int i = 0; i < 15; i++) printf(" %.6f", values[i]);
	   printf(" ...\n");
	 */
}
void pr_experiments(
		const Graph * const g,
		int size
		)
{
	printf("-------------------------------------------------------------------\n");
	//reverse the graph
	const Graph * r =get_reverse_graph(g);
	// partition on the Graph
	printf("Partitioning ... ");
	timer_start();
	struct part_table * t =
		partition(
				r->vertex_num,
				r->edge_num,
				r->vertex_begin,
				r->edge_dest,
				PAR_NUM 
				);
	if (t == NULL) {
		perror("Partition Failed !");
		exit(1);
	}else{
		printf("%.2f ms", timer_stop());
	}
	// get Partitions of reversed graph
	printf("Get partitions  ... ");
	timer_start();
	Graph ** part_r = get_cut_graphs(r, t);
	if(part_r == NULL){
		perror("Failed !");
		exit(1);
	}else{
		printf("%.2f ms\n", timer_stop());
	}


	//test pr
	float  *value_cpu =(float *)calloc(g->vertex_num,sizeof(float));
	float  *value_gpu = (float *)calloc(g->vertex_num,sizeof(float));
	int * out_degree = (int *)calloc(g->vertex_num,sizeof(int));
	int * in_degree = (int *)calloc(g->vertex_num,sizeof(int));
	if(value_cpu==NULL || value_gpu==NULL)
	{
		perror("Out of Memory for values");
		exit(1);
	}

	//get degree
	get_degree(g,in_degree,out_degree);
	printf("\tTime\tTotal\tTips\n");

	//cpu_pagerank(g,value_cpu);
	/*
	   printf("\n");
	   gpu_pr_navie(part_r,t,out_degree, value_gpu);
	   check_values(value_gpu,value_cpu,g->vertex_num);

	 */
	gpu_pr_atomic(part_r, t,out_degree, value_gpu,size);
	//check_values(value_gpu,value_cpu,g->vertex_num);
	release_table(t);
	for (int i = 0; i < PAR_NUM ; i++) release_graph(part_r[i]);
	free(part_r);
	free(value_cpu);
	free(value_gpu);

}
