#include <stdio.h>
#include <string>
#include <algorithm>
#include <math.h>
#include "../preprocessing/PreProcessing.hpp"
#include "../gpuengine/InitGPUDevice.h"
#include "sm_12_atomic_functions.h"

using namespace std;
#define DEV_ID 0
#define ROOT_ID 5
#define INIT_VAL 1024
#define PG_VAL 1.0000
#define PG_D 0.5
#define PG_D_STOP 0.01
#define BFS_CHUNK 4 
#define W_SE 4
#define CO_CHUNK (10-1)


__global__ void BFS_V_S(unsigned int *values, struct VertexData *vertex, struct EdgeData *edges, unsigned int v_size, int step, int *stop)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % v_size;
	unsigned int v_id = vertex[index].id;

	if(values[v_id] == step)
	{
		unsigned int v_b = vertex[index].begin, v_e = vertex[index].end;
		for(unsigned int i = v_b; i < v_e; i++)
		{
			if(values[edges[i].dst] == INIT_VAL)
			{
				values[edges[i].dst] = step + 1;	
				stop[0] = 1;
			}
		}
	}
}

__global__ void BFS_V_D(unsigned int *values, struct VertexData *vertex, struct EdgeData *edges, unsigned int v_size, int step, int *stop)
{
        int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % v_size;
        unsigned int v_id = vertex[index].id;

        if(values[v_id] == INIT_VAL)
        {
                unsigned int v_b = vertex[index].begin, v_e = vertex[index].end, min_val = INIT_VAL;
                for(unsigned int i = v_b; i < v_e; i++)
                {
                        unsigned int val_src = values[edges[i].src];

                        if(val_src < min_val)
                        {
                                min_val = val_src;
                        }
                }
                if(INIT_VAL != min_val)
                {
                        values[v_id] = min_val + 1;
                        stop[0] = 1;
                }
        }
}

__global__ void BFS_E(unsigned int *values, struct EdgeData *edges, unsigned int e_size, int step, int *stop)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % e_size;
	unsigned int indexSrc = edges[index].src, indexDst = edges[index].dst;

	if(values[indexSrc] == step)
        {
                if(values[indexDst] == INIT_VAL)
                {
                        values[indexDst] = step + 1;
                        stop[0] = 1;
                }
        }

}

__global__ void BFS_E_W(unsigned int *values, struct EdgeData *edges, unsigned int e_size, int step, int *stop)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % e_size;
    unsigned int indexSrc = edges[index].src, indexDst = edges[index].dst;

    if(values[indexSrc] == step)
    {
        if(values[indexDst] == INIT_VAL)
        {
            values[indexDst] = step + 1;
            stop[0] = 1;
        }
    }

}

__global__ void PageRank_V_D(float *values, struct VertexData *vertex, struct EdgeData *edges, unsigned int *degrees, unsigned int v_size, unsigned int *stop, int step, int *iter)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % v_size;
        unsigned int v_id = vertex[index].id, v_b = vertex[index].begin, v_e = vertex[index].end, indexSrc;
	float sum = 0, old = values[v_id];

	if(step == iter[0])
	{
		int idx1 = threadIdx.x / W_SE + blockIdx.x * blockDim.x;
		index = idx1 % v_size;
        	v_id = vertex[index].id;
		v_b = vertex[index].begin;
		v_e = vertex[index].end;
		old = values[v_id];

                for(unsigned int i = v_b + threadIdx.x % W_SE; i < v_e; i += W_SE)
        	    {
                	indexSrc = edges[i].src;
			        float a_val = atomicAdd(&values[indexSrc], 0.0f);
                	sum = sum + a_val / ( degrees[indexSrc] * 1.0);
                }
		float tmp = PG_D * sum + 1 - PG_D;
		atomicExch(&values[v_id], tmp);

		if(fabsf(old - values[v_id]) > PG_D_STOP)
        	{
            		stop[0] = 1;
		}
		return;
	}

	{
        	for(unsigned int i = v_b; i < v_e; i++)
        	{
			indexSrc = edges[i].src;
			sum = sum + values[indexSrc] / ( degrees[indexSrc] * 1.0);
        	}

		values[v_id] = PG_D * sum + 1 - PG_D;

		if(fabsf(old - values[v_id]) > PG_D_STOP)
		{
			//atomicSub(stop, 1);
			stop[0] = 1;
		}
	}
}

__global__ void Component_V_D(unsigned int *values, struct VertexData *vertex, struct EdgeData *edges, unsigned int v_size, int step, int *stop)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % v_size;
	unsigned int v_id = vertex[index].id;

	unsigned int v_b = vertex[index].begin, v_e = vertex[index].end, level = values[v_id];
	for(unsigned int i = v_b; i < v_e; i++)
	{
		if(values[edges[i].src] < level)
		{
			level = values[edges[i].src];
		}
	}

	if(level != values[v_id])
	{
		for(unsigned int i = v_b; i < v_e; i++)
		{
			values[edges[i].src] = level;
		}
		values[v_id] = level;
		stop[0] = 1;
	}
}

__global__ void Component_V(unsigned int *values, struct VertexData *vertex, struct EdgeData *edges, unsigned int v_size, int *stop)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % v_size;
        unsigned int v_id = vertex[index].id, v_b = vertex[index].begin, v_e = vertex[index].end, indexSrc, level = values[v_id];

	for(unsigned int i = v_b; i < v_e; i++)
	{
		if(values[edges[i].dst] < level)
		{
			level = values[edges[i].dst];
		}
	}

	if(level != values[v_id])
        {
                for(unsigned int i = v_b; i < v_e; i++)
                {
                        values[edges[i].dst] = level;
                }
                values[v_id] = level;
                stop[0] = 1;
        }
	
}


__global__ void Component_E(unsigned int *values, struct EdgeData *edges, unsigned int e_size, int step, int *stop)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % e_size;
        unsigned int indexSrc = edges[index].src, indexDst = edges[index].dst;
	unsigned int valSrc = values[indexSrc], valDst = values[indexDst], tmpval;

	if(valSrc > valDst)
	{
		values[indexSrc] = valDst;
		stop[0] = 1;
		return ;
	}
/*    
	if(valDst > valSrc)
	{
		values[indexDst] = valSrc;
		stop[0] = 1;
		return ;
	}
*/
}

__global__ void dijkstra_first_phase_e(unsigned int *value, unsigned int *new_value, struct EdgeData *edges, unsigned int e_size, bool *to_update)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x, index = idx % e_size;
        unsigned int indexSrc = edges[index].src, indexDst = edges[index].dst;
	
        if(to_update[indexSrc])
        {
		unsigned int new_weight = value[indexSrc] + edges[index].weight;
		if(new_weight < new_value[indexDst])
		{
			new_value[indexDst] = new_weight;
		}
        }

	return ;
}

__global__ void dijkstra_first_phase_v(unsigned int *value, unsigned int *new_value, struct VertexData *vertex, struct EdgeData *edges, unsigned int v_size, bool *to_update)
{
        int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index > v_size)
	{
		return ;
	}

        unsigned int vertex_id = vertex[index].id, e_begin = vertex[index].begin, e_end = vertex[index].end;

        if(to_update[vertex_id])
        {
		for(unsigned int i = e_begin; i < e_end; i++)
		{
                	unsigned int new_weight = value[vertex_id] + edges[i].weight;
			//tests only, worng value
			new_weight = value[vertex_id] + 2;
                	if(new_weight < new_value[edges[i].dst])
                	{
                        	new_value[edges[i].dst] = new_weight;
                	}
		}
        }

        return ;
}


__global__ void dijkstra_second_phase(unsigned int *value, unsigned int *new_value, unsigned int v_size, bool *to_update, int *stop)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index > v_size)
	{
		return ;
	}

	unsigned int vertex_id = index;

	if (new_value[vertex_id] < value[vertex_id]) {
    		value[vertex_id] = new_value[vertex_id];
    		to_update[vertex_id] = true;
    		stop[0] = 1;
  	}
  	new_value[vertex_id] = value[vertex_id];
}
