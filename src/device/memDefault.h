#pragma once
#include "../graphdata/EdgeData.hpp"
#include "../gpuengine/initGPU.h"

/*values mem malloc #0
 *edge num record for each chunk #1
 *edgedata mem malloc #2
 *vertex num record for each chunk #3
 *vertexdata mem malloc #4
 *degree mem malloc , for pagerank #5
 *lock mem for the last partition #6
 *lock count for the last partition #7
 *lock bool, to make sure start from any index #8
 *
 */


//values mem malloc #0
#ifdef PG_VAL_TYPE
typedef float V_TYPE;
typedef float* V_P_TYPE;
#endif

#ifdef BFS_VAL_TYPE
typedef unsigned int V_TYPE;
typedef unsigned int* V_P_TYPE;
#endif

V_P_TYPE m_value;
V_P_TYPE d_value;
unsigned int m_size;//vertex size

int mem_vertex_values(unsigned int v_num)
{
	m_size = v_num + 1;
	m_value = (V_P_TYPE)malloc(sizeof(V_TYPE) * m_size);
	memset(m_value, 0, sizeof(V_TYPE) / sizeof(V_TYPE) * m_size);
	std::cout << "Malloc vertex values size " << m_size << std::endl;
	return m_size;
}

int release_mem_values()
{
	free(m_value);
	m_size = 0;
	return m_size;
}

int d_mem_vertex_values(unsigned int d_size)
{
	cudaMalloc((void **)&d_value, sizeof(V_TYPE) * (d_size + 1));
	return d_size;
}

int d_release_mem_values()
{
	cudaFree(d_value);
	return 0;
}
int memcpy_value_m2d()
{
	cudaMemcpy(d_value, m_value, sizeof(V_TYPE) * m_size, cudaMemcpyHostToDevice);
	return m_size;
}

int memcpy_value_d2m()
{
	cudaMemcpy(m_value, d_value, sizeof(V_TYPE) * m_size, cudaMemcpyDeviceToHost);
        return m_size;
}

// edge num record for each chunk #1
unsigned int *m_edge;// by src order
unsigned int *n_edge;// by dst order
unsigned int *num; // to point src or dst order, 0 for src, 1 for dst
int src_or_dst;
int set_n_edge(int _chunk, int _order)
{
	int _size = _chunk;
	src_or_dst = _order;
        n_edge = (unsigned int *)malloc(sizeof(unsigned int) * _size);
	m_edge = (unsigned int *)malloc(sizeof(unsigned int) * _size);
        memset(n_edge, 0, sizeof(unsigned int) / sizeof(char) * _size);
	memset(m_edge, 0, sizeof(unsigned int) / sizeof(char) * _size);

	if(src_or_dst == 0)
        {
                num = m_edge;
        }
        else
        {
                num = n_edge;
        }

        std::cout << "Set edges num of each chunk......" << std::endl;
        return _size;
}

void release_n_edge()
{
        free(n_edge);
	free(m_edge);
}

//edgedata for each chunk mem malloc #2
struct EdgeData **m_all_edge;
struct EdgeData **d_all_edge;
unsigned int e_size;//colord edge num
int e_chunk;

int mem_all_edges(int _chunk)
{
	e_chunk = _chunk;

	std::cout << "Starting malloc edges mem......." << std::endl;
        m_all_edge = (struct EdgeData **) malloc(sizeof(struct EdgeData *) * e_chunk);
	for(int n = 0; n < e_chunk; n++)
	{
		std::cout << "edge malloc chunk = " << n << " , size = " << num[n] << std::endl;
		m_all_edge[n] = (struct EdgeData *) malloc(sizeof(struct EdgeData) * num[n]);
	}
        return e_chunk;
}

int d_mem_all_edges(int _chunk)
{
	std::cout << "Starting malloc edges gpu mem......." << std::endl;
	d_all_edge = (struct EdgeData **) malloc(sizeof(struct EdgeData *) * _chunk);
	for(int n = 0; n < _chunk; n++)
	{
		//std::cout << "gpu edge malloc chunk = " << n << " , size = " << num[n] << std::endl;
		cudaError_t cuda_edge = cudaMalloc((void **)&d_all_edge[n], sizeof(struct EdgeData) * num[n]);
		if(cuda_edge == cudaSuccess)
		{
			std::cout << "gpu edge malloc chunk = " << n << " , size = " << num[n] << std::endl;
			//std::cout << "Error......malloc device mem[ " << n << " ] failed......" << std::endl;
		}
	}
	return _chunk;
}

int release_all_edges()
{
	for(int n = 0; n < e_chunk; n++)
        {
		free(m_all_edge[n]);
        }
        free(m_all_edge);
        e_chunk = 0;
        return e_chunk;
}

int release_d_all_edges()
{
	for(int n = 0; n < e_chunk; n++)
	{
		cudaFree(d_all_edge[n]);
	}
	free(d_all_edge);
	return 0;
}

int memcpy_all_edges_m2d(int _chunk)
{
        int _d_size = num[_chunk];
        cudaMemcpy(d_all_edge[_chunk], m_all_edge[_chunk], sizeof(struct EdgeData) * _d_size, cudaMemcpyHostToDevice);
        return m_size;
}

int memcpy_all_edges_d2m(int _chunk)
{

        int _d_size = num[_chunk];
        cudaMemcpy(m_all_edge[_chunk], d_all_edge[_chunk], sizeof(struct EdgeData) * _d_size, cudaMemcpyDeviceToHost);
        return m_size;
}

//vertex num record for each chunk #3
unsigned int *n_vertex;
unsigned int *d_vertex;
int set_n_vertex(int _chunk)
{
	int _size = _chunk;
        n_vertex = (unsigned int *)malloc(sizeof(unsigned int) * _size);
        memset(n_vertex, 0, sizeof(unsigned int) / sizeof(char) * _size);

	cudaMalloc((void **)&d_vertex, sizeof(unsigned int) * _size);
        std::cout << "Set vertex num of each chunk......" << std::endl;
        return _size;
}

void release_n_vertex()
{
        free(n_vertex);
	cudaFree(d_vertex);
}

int memcpy_vertex_m2d(int chunk)
{
	cudaMemcpy(d_vertex, n_vertex, sizeof(unsigned int) * chunk, cudaMemcpyHostToDevice);
	return chunk;
}

//vertexdata for each chunk mem malloc #4
struct VertexData **m_all_vertex;
struct VertexData **d_all_vertex;

int mem_all_vertex(int _chunk)
{
        int v_chunk = _chunk;
        std::cout << "Starting malloc vertex mem......." << std::endl;
        m_all_vertex = (struct VertexData **) malloc(sizeof(struct VertexData *) * v_chunk);
        for(int n = 0; n < v_chunk; n++)
        {
                std::cout << "vertex malloc chunk = " << n << " , size = " << n_vertex[n] << std::endl;
                m_all_vertex[n] = (struct VertexData *) malloc(sizeof(struct VertexData) * n_vertex[n]);
        }
        return e_chunk;
}

int d_mem_all_vertex(int _chunk)
{
        std::cout << "Starting malloc vertex gpu mem......." << std::endl;
        d_all_vertex = (struct VertexData **) malloc(sizeof(struct VertexData *) * _chunk);
        for(int n = 0; n < _chunk; n++)
        {
                cudaError_t cuda_vertex = cudaMalloc((void **)&d_all_vertex[n], sizeof(struct VertexData) * n_vertex[n]);
                if(cuda_vertex == cudaSuccess)
                {
                        std::cout << "gpu vertex malloc chunk = " << n << " , size = " << n_vertex[n] << std::endl;
                }
        }
        return _chunk;
}

int release_all_vertex()
{
        for(int n = 0; n < e_chunk; n++)
        {
                free(m_all_vertex[n]);
        }
        free(m_all_vertex);
        e_chunk = 0;
        return e_chunk;
}

int release_d_all_vertex()
{
        for(int n = 0; n < e_chunk; n++)
        {
                cudaFree(d_all_vertex[n]);
        }
        free(d_all_vertex);
        return 0;
}

int memcpy_all_vertex_m2d(int _chunk)
{
        int _d_size = n_vertex[_chunk];
        cudaMemcpy(d_all_vertex[_chunk], m_all_vertex[_chunk], sizeof(struct VertexData) * _d_size, cudaMemcpyHostToDevice);
        return m_size;
}

int memcpy_all_vertex_d2m(int _chunk)
{

        int _d_size = n_vertex[_chunk];
        cudaMemcpy(m_all_vertex[_chunk], d_all_vertex[_chunk], sizeof(struct VertexData) * _d_size, cudaMemcpyDeviceToHost);
        return m_size;
}

int set_vertex_data_val(int chunk)
{
        for(unsigned int n = 0; n < n_vertex[chunk]; n++)
        {
                m_all_vertex[chunk][n].val = m_value[m_all_vertex[chunk][n].id];
        }
        return chunk;
}

//degree mem malloc , for pagerank #5
unsigned int *m_degree;
unsigned int *d_degree;
int mem_out_degree(unsigned int _size)
{
        unsigned int size = _size + 1;
        m_degree = (unsigned int *)malloc(sizeof(unsigned int) * size);
        memset(m_degree, 0, sizeof(unsigned int) / sizeof(char) * size);
        std::cout << "Malloc vertex out degree size " << size << std::endl;
        return size;
}

int release_mem_out_degree()
{
        free(m_degree);
        m_size = 0;
        return m_size;
}

int d_mem_out_degree(unsigned int d_size)
{
        unsigned int size = d_size + 1;
        cudaMalloc((void **)&d_degree, sizeof(unsigned int) * size);
        return size;
}

int d_release_out_degree()
{
        cudaFree(d_degree);
        return 0;
}

int memcpy_degree_m2d(unsigned int d_size)
{
	unsigned int size = d_size + 1;
        cudaMemcpy(d_degree, m_degree, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
        return m_size;
}

//lock mem for the last partition #6
unsigned int *v_lock;
unsigned int *d_lock;

int mem_vertex_lock(unsigned int size)
{
	v_lock = (unsigned int *)malloc(sizeof(unsigned int) * size);
	return size;
}

int d_mem_vertex_lock(unsigned int size)
{
	cudaMalloc((void **)&d_lock, sizeof(unsigned int) * size);
	return size;
}

int release_vertex_lock()
{
	free(v_lock);
	return 0;
}

int d_release_vertex_lock()
{
	cudaFree(d_lock);
	return 0;
}

int memcpy_lock_m2d(unsigned int size)
{
	cudaMemcpy(d_lock, v_lock, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
	return size;
}


//lock count for the last partition #7
unsigned int *count_lock;
unsigned int *d_count_lock;

int mem_count_lock(unsigned int size)
{
        count_lock = (unsigned int *)malloc(sizeof(unsigned int) * size);
	memset(count_lock, 0, sizeof(unsigned int ) / sizeof(char) * size);
        return size;
}

int d_mem_count_lock(unsigned int size)
{
        cudaMalloc((void **)&d_count_lock, sizeof(unsigned int) * size);
        return size;
}

int release_count_lock()
{
        free(count_lock);
        return 0;
}

int d_release_count_lock()
{
        cudaFree(d_count_lock);
        return 0;
}

int memcpy_count_lock_m2d(unsigned int size)
{
        cudaMemcpy(d_count_lock, count_lock, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
        return size;
}


//lock bool, to make sure start from any index #8
unsigned int *bool_lock;
unsigned int *d_bool_lock;

int mem_bool_lock(unsigned int size)
{
        bool_lock = (unsigned int *)malloc(sizeof(unsigned int) * size);
        memset(bool_lock, 0, sizeof(unsigned int ) / sizeof(char) * size);
        return size;
}

int d_mem_bool_lock(unsigned int size)
{
        cudaMalloc((void **)&d_bool_lock, sizeof(unsigned int) * size);
        return size;
}

int release_bool_lock()
{
        free(bool_lock);
        return 0;
}

int d_release_bool_lock()
{
        cudaFree(d_bool_lock);
        return 0;
}

int memcpy_bool_lock_m2d(unsigned int size)
{
        cudaMemcpy(d_bool_lock, bool_lock, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);
        return size;
}
