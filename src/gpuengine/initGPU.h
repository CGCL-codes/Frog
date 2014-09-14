#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

int num_gpu_dev;
int current_dev;

unsigned int d_stop;


int get_device_count()
{
        int device_num = -1;
        cudaGetDeviceCount(&device_num);
	num_gpu_dev = device_num;
	std::cout << "the total gpu number is " << device_num << std::endl;
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

