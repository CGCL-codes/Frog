#define CC_MEM_H
#define CC_VAL_TYPE
#include <stdio.h>
#include <stdio.h>
#include <string>
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

	PreProcessing *pre = new PreProcessing;

	pre->get_partition(file_name, chunk);
	
	set_n_edge(chunk, 0);
	unsigned int e_num = pre->get_all_edges(file_name, chunk);
	unsigned int v_num = pre->vv.get_vertex_num();

	for(int j = 0; j < chunk; j++)
        {
                std::cout<< "n_edge[ " << j << " ] = " << num[j] << std::endl;
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

	//mem edges for each chunk
	mem_all_edges(chunk);
	std::cout << "mem_all_edges ... " <<std::endl;
	pre->set_partition_edges_src(chunk, e_num);
    printf("set_partition_edges_src success......\n");

    //d_mem_max_edges(chunk);
    d_mem_all_edges(chunk);

	mem_all_vertex(chunk);
	set_partition_vertex_src(chunk);
	d_mem_all_vertex(chunk);

    std::cout << "deleting preprocessing in main ..." << std::endl;
    delete pre;

	cudaEvent_t start, stop, start1, stop1, start2, stop2, start_cc, stop_cc;
	float gpu_time_excute = 0, gpu_time = 0, gpu_time_edge = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
	cudaEventCreate(&start_cc);
        cudaEventCreate(&stop_cc);

	//values mem
	mem_vertex_values(v_num);
        for(unsigned int v = 0; v < m_size; v++)
        {
                m_value[v] = v;
        }
        //m_value[ROOT_ID] = 0;
	d_mem_vertex_values(v_num);
	memcpy_value_m2d(0);

	for(int i = 0; i < chunk; i++)
	{
		memcpy_all_vertex_m2d(i, i);
	}
    for(int i = 0; i < chunk; i++)
    {
        memcpy_all_edges_m2d(i, i);
    }

	std::cout << "Starting GPU processing and timeing...." << std::endl;

	int *m_stop = (int *)malloc(sizeof(int) * 1), *d_stop;
	cudaMalloc((void **)&d_stop, sizeof(int) * 1);
	m_stop[0] = 1;
	cudaEventRecord(start1, 0);
	//for(int i = 0; i < iter && ; i++)
	int iter = 30, i = 0, j;
	while(m_stop[0])
	{
		m_stop[0] = 0;
		cudaMemcpy(d_stop, m_stop, sizeof(int) * 1, cudaMemcpyHostToDevice);
		//cudaMemset(d_stop, 0, sizeof(int));
		//int j;
		for(j = 0; j < chunk; j++)
		{
			int blockv = (num[j] + 256 - 1) / 256;
            //memcpy_chunk_edges_m2d(j);
			cudaEventRecord(start_cc, 0);
			Component_E<<<blockv, 256>>>(d_value, d_all_edge[j], num[j], 0, d_stop);
			//cudaDeviceSynchronize();
            cudaEventRecord(stop_cc, 0);
            cudaEventSynchronize(stop_cc);
            float tmp_time;
            cudaEventElapsedTime(&tmp_time, start_cc, stop_cc);
            gpu_time_excute += tmp_time;
		}
		i++;
		cudaMemcpy(m_stop, d_stop, sizeof(int) * 1, cudaMemcpyDeviceToHost);
	}

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&gpu_time, start1, stop1);

	std::cout <<"Total GPU proc time is: " << gpu_time_excute << std::endl;

	memcpy_value_d2m(0);

	for (int i = 0; i < chunk; i++)
        {
                cudaStreamSynchronize(stream[i]);
        }
        for (int i = 0; i < chunk; i++)
        {
                cudaStreamDestroy(stream[i]);
        }
	
	std::set<unsigned int> result;
        for(unsigned int v = 0; v < m_size; v++)
        {
                result.insert(m_value[v]);
	}

	unsigned int *l_size, tmp_size = result.size();
        l_size = (unsigned int *) malloc(sizeof(unsigned int) * tmp_size);
        memset(l_size, 0, sizeof(unsigned int) / sizeof(char) * tmp_size);

        for(unsigned int v = 0; v < v_num; v++)
        {
                l_size[m_value[v] % tmp_size]++;
        }

        for(unsigned int i = 0; i < 20; i++)
        {
            std::cout << "level " << i << " has vertices : " << l_size[i] << std::endl;
		    //std::cout << "value " << i << " is " << m_value[i] << std::endl;
        }
        free(l_size);
        std::cout << "Result is component = " << tmp_size << std::endl;

	release_n_edge();
	release_mem_values();
    	d_release_mem_values();
	release_all_edges();
	release_d_max_edges();

	std::cout << "......Preprossing Complete !" << std::endl;

	return 1;
}
