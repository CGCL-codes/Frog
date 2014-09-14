#include <iostream>
#include <cuda_runtime.h>

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

        std::cout << "Table Create Completed !" << std::endl;

	int *d_vertex, *	
}
