#define PG_MEM_H
#define PG_VAL_TYPE
#include <stdio.h>
#include <stdio.h>
#include <string>
#include "../preprocessing/PreProcessing.hpp"
//#include "excuteGPU.h"
//#include "prepartitionGPU.hpp"
#include "../preprocessing/hybrid_vertices.hpp"
//#include "InitGPUDevice.h"

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
	
	//pre.debug_print_vertex(chunk);

	set_n_edge(chunk, 1);
	unsigned int e_num = pre->get_all_edges(file_name, chunk);
	unsigned int v_num = pre->vv.get_vertex_num();

	for(int j = 0; j < chunk; j++)
	{
		std::cout<< "n_edge[" << j << "] = " << pre->vv.vvSize[j] << " ,m_edge[" << j << "] = " << num[j] << std::endl;
	}

	//pre->get_n_vertex_partition(chunk);
	mem_all_edges(chunk);
	pre->set_partition_edges_dst(chunk, e_num);
	
        delete pre;

	for(int i = 0; i < chunk; i++)
	{
		n_vertices_partition(i);
	}

	for(int j = 0; j < chunk; j++)
	{
		for(int i = 0; i < chunk; i++)
		{
			std::cout << "vertices between " << j << " && " << i << " = " << n_hybird_vertices(j, i, v_num) << std::endl;
			//std::cout << "vertices between " << i << " && " << i+1 << " = " << n_hybird_vertices(i, i+1, v_num) << std::endl;
		}
		std::cout << std::endl;
	}

	//std::cout << "Result is vertex = "<< max_v << " , max = " << maxval << std::endl;

	release_n_edge();
	release_all_edges();

/*
	int i = 0, total = 0;
	while(i < chunk)
	{
		std::cout << "processing chunk num is " << i << std::endl;
		total += pre.get_chunk_edges(i++, file_name);
	}

	std::cout << "Total edge num is " << total <<std::endl;
*/
	std::cout << "......Preprossing Complete !" << std::endl;

	//std::cout << "Init GPU Device..." << std::endl;

	//int device_num = get_device_count();

	return 1;
}
