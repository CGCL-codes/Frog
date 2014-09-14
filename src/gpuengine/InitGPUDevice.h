#include <cuda_runtime.h>
#include <iostream>
#include <string>

class InitDevice{
public:
	int num_gpu_dev;
	int current_dev;
	int chunk_num;

	cudaDeviceProp device_prop;
	cudaStream_t *sync_stream;
public:
	InitDevice(int _chunk_num, int _use_stream):chunk_num(_chunk_num)
	{
		sync_stream = NULL;
		if(_use_stream == 1 && sync_stream == NULL)
		{
			sync_stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * chunk_num);
		}
	}

	~InitDevice()
	{
		if(sync_stream != NULL)
		{
			free(sync_stream);
		}
	}

public:
	int get_device_count()
	{
        	int device_num = -1;
        	cudaGetDeviceCount(&device_num);
		num_gpu_dev = device_num;
        	return num_gpu_dev;
	}

	int set_current_device(int dev_id)
	{
		current_dev = dev_id;
		if(cudaSetDevice(dev_id)!=cudaSuccess)
		{
			std::cout << "use device failed ..., the dev id is " << dev_id << std::endl;
			return 0;
		}
		return current_dev;
	}

	int get_device_id()
	{
		int dev_id;
		if(cudaGetDevice(&dev_id)!=cudaSuccess)
		{
			std::cout << "get device failed ..." << std::endl;
		}

		return dev_id;
	}

	int init_device()
	{
		return 0;	
	}
	
	int create_dev_stream()
	{
		return 0;
	}
};
