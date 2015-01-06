#define PG_S_MEM_H
#define PG_S_VAL_TYPE
#include <stdio.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <sys/time.h>
#include "../preprocessing/PreProcessing.hpp"
#include "../gpuengine/excuteGPU.h"
#include "../preprocessing/prepartitionGPU.hpp"

using namespace std;

int main(int argc, char *argv[])
{
	if(argc < 3)
	{
		std::cout << "main input error." << std::endl;
		exit(-1);
	}

	std::string file_name = argv[1];
	int chunk = atoi(argv[2]);

	std::cout << "sizeof unsigned int = " << sizeof(unsigned int) / sizeof(char) << " B." << std::endl;

	PreProcessing *pre = new PreProcessing;

	struct timeval start_pre, end_pre;
	long start_time = 0, end_time = 0;
	gettimeofday(&start_pre, NULL);
	start_time = ((long)start_pre.tv_sec)*1000 +  (long)start_pre.tv_usec/1000;
	pre->get_partition(file_name, chunk);

	gettimeofday(&end_pre, NULL);
        end_time = ((long)end_pre.tv_sec)*1000 +  (long)end_pre.tv_usec/1000;

	std::cout << "time pre-processing : " << end_time - start_time << " ms." << std::endl;
	
	set_n_edge(chunk, 1);
	unsigned int e_num = pre->get_all_edges(file_name, chunk);
	unsigned int v_num = pre->vv.get_vertex_num();

	for(int j = 0; j < chunk; j++)
	{
		std::cout<< "m_edge[ " << j << " ] = " << num[j] << std::endl;
	}

	init_cuda_stream(chunk);
	for (int i = 0; i < chunk; i++)
        {
                cudaStreamCreate(&stream[i]);
        }

	pre->get_n_vertex_partition(chunk);
	memcpy_vertex_m2d(chunk, 0);


    std::cout << "vv.size = " << pre->vv.vv.size();
    {
        std::set<unsigned int> temp;
        temp.swap(pre->vv.vv);
        if(pre->vv.vvArray != NULL)
        {
            free(pre->vv.vvArray);
        }
    }

    std::cout << ", free vertexvector size = " << pre->vv.vv.size() << ", vvArray = " << sizeof(pre->vv.vvArray) << std::endl;


	mem_all_edges(chunk);
	pre->set_partition_edges_dst(chunk, e_num);
	printf("set_partition_edges_dst success......\n");
    //all graphs
    //d_mem_max_edges(chunk);

    //for small graphs only, cannot work for large graph like twitter
    d_mem_all_edges(chunk);

	std::cout << "Compute the out degrees......" << std::endl;
        mem_out_degree(v_num);

	for(int i = 0; i < chunk; i++)
	{
		for(unsigned int n = 0; n < num[i]; n++)
		{
			m_degree[m_all_edge[i][n].src]++;
		}
	}
        d_mem_out_degree(v_num);
        memcpy_degree_m2d(v_num, 1);	

	std::string file_degree = file_name + ".degree";
	FILE *fp_degree = fopen(file_degree.c_str(), "w+");
	unsigned int max_degree = 0;
	fprintf(fp_degree, "v %u\n", v_num);
	for(unsigned int n = 0; n < v_num+1; n++)
	{
		//vertex_id, chunk, degree
		if(m_degree[n] > max_degree) max_degree = m_degree[n];
		fprintf(fp_degree, "[%d, %u],\n",pre->vv.vvPartition[n], m_degree[n]);
	}
	std::cout << "max degree is " << max_degree << std::endl;
	fclose(fp_degree);
	
    mem_all_vertex(chunk);
	set_partition_vertex_dst(chunk);

	d_mem_all_vertex(chunk);

    std::cout << "deleting preprocessing in main ..." << std::endl;
    delete pre;

	cudaEvent_t start, stop, start1, stop1, start2, stop2;
	float gpu_time_excute = 0, gpu_time = 0, gpu_time_edge = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
        cudaEventCreate(&stop2);

	mem_vertex_values(v_num);
        for(unsigned int v = 0; v < m_size; v++)
        {
                m_value[v] = PG_VAL;
        }
        std::cout << "init m_value success..." << std::endl;
	d_mem_vertex_values(v_num);
	memcpy_value_m2d(0);
/*
	for(int i = 0; i < chunk; i++)
	{
		set_vertex_data_val(i);
	}
*/

	std::cout << "Starting GPU processing and timeing...." << std::endl;

	int iter = 10;
	unsigned int *m_stop = (unsigned int *)malloc(sizeof(unsigned int) * 1), *d_stop;
	int *m_iter, *d_iter;
	cudaMallocHost(&m_iter, sizeof(int));
        cudaMalloc((void **)&d_iter, sizeof(int) * 1);

        cudaMalloc((void **)&d_stop, sizeof(unsigned int) * 1);
	m_stop[0] = 1;
	m_iter[0] = chunk - 1;
	cudaMemcpyAsync(d_iter, m_iter, sizeof(int) * 1, cudaMemcpyHostToDevice, stream[1]);
	cudaEventRecord(start1, 0);

	//little graph only TODO
	for(int j = 0; j < chunk; j++)
	{
		memcpy_all_vertex_m2d(j, j);
	}

    for(int j = 0; j < chunk; j++)
    {
        memcpy_all_edges_m2d(j, j);
    }

	unsigned int step_tmp = 0;
	for(int i = 0; i < iter; i++)
	{
		m_stop[0] = 0;
                cudaMemcpy(d_stop, m_stop, sizeof(unsigned int) * 1, cudaMemcpyHostToDevice);

		for(int j = 0; j < chunk; j++)
		{
			int blockx = (n_vertex[j] + 256 - 1 ) / 256;
			//memcpy_chunk_edges_m2d(j);
			PageRank_V_D<<<blockx, 256, 0 , stream[j]>>>(d_value, d_all_vertex[j], d_all_edge[j], d_degree, n_vertex[j], d_stop, j, d_iter);
		} 
		cudaMemcpy(m_stop, d_stop, sizeof(unsigned int) * 1, cudaMemcpyDeviceToHost);
		std::cout << "iter = " << i << std::endl;
	}

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&gpu_time, start1, stop1);

	std::cout <<"Total proc time is: " << gpu_time << " ms" <<std::endl;

	memcpy_value_d2m(0);

	for (int i = 0; i < chunk; i++)
        {
                cudaStreamSynchronize(stream[i]);
        }
	for (int i = 0; i < chunk; i++)
        {
                cudaStreamDestroy(stream[i]);
        }

	//unsigned int step = 0;
	float maxval = 0;
	double sumval = 0;
	unsigned int max_v = 0;
	for(unsigned int v = 0; v < v_num; v++)
	{
		//if(v % 10000 == 0)
		{
			if(m_value[v] > maxval)
			{
				maxval = m_value[v];
				max_v = v;
			}
			//std::cout << v <<" vertex pagerank value is " << m_value[v] << std::endl;
		}
		if(v < 10)
		{
			std::cout << v <<" vertex pagerank value is " << m_value[v] << std::endl;
		}
		sumval += m_value[v];
	}

	std::cout << "Result is vertex = "<< max_v << " , max = " << maxval << " , sum = " << sumval << std::endl;

	release_n_edge();
	release_mem_values();
        d_release_mem_values();
	release_mem_out_degree();
	d_release_out_degree();
	release_all_edges();
	//release_d_max_edges();
    release_d_all_edges();


	std::cout << "......Preprossing Complete !" << std::endl;

	return 0;
}
