#include <stdio.h>
#include <stdio.h>
#include <string>
#include "../preprocessing/PreProcessing.hpp"
//#include "InitGPUDevice.h"
#include "../gpuengine/initGPU.h"

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

	PreProcessing pre;
	pre.get_partition(file_name, chunk);
	
	pre.debug_print_vertex(chunk);

	//pre.get_chunk_vv(file_name, chunk-1, chunk);
	//pre.debug_print_chunk(chunk-1);

	pre.get_all_edges(file_name);

	//std::cout << "The total number of edges is " << pre.ev.get_edge_num();

	
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

	std::cout << "Init GPU Device..." << std::endl;

	int device_num = get_device_count();

	if(device_num)
	{
		set_current_device(0);
		std::cout << "Set the gpu to be used : " << get_device_id() << std::endl;
	}
}
