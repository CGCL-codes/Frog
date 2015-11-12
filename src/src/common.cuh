#ifndef FROG_COMMON_H_INCLUDED
#define FROG_COMMON_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#ifdef __CUDA_RUNTIME_H__
#define HANDLE_ERROR(err) if (err != cudaSuccess) {	\
            printf("CUDA Error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError()));\
            exit(1);\
        }
#endif  // #ifdef __CUDA_RUNTIME_H__

class auto_utility
{
private:
    // pointers to be released when ~() is called
    std::vector<void *> host_pointers;
#ifdef __CUDA_RUNTIME_H__
    std::vector<void *> dev_pointers;
    cudaEvent_t start, stop;
#endif  // #ifdef __CUDA_RUNTIME_H__
public:
#ifdef __CUDA_RUNTIME_H__
    auto_utility() {
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
    }
#endif  // #ifdef __CUDA_RUNTIME_H__
    void * _Malloc(size_t size, const char * FILE, const int LINE) {
        void * ret = malloc(size);
        if (ret == NULL) {
            printf("Memory Allocation Failed in File '%s' at Line %d!\n", FILE, LINE);
            perror("\tINFO");
            exit(1);
        }
        host_pointers.push_back(ret);
        return ret;
    }
    void * _Calloc(size_t num, size_t size, const char * FILE, const int LINE) {
        void * ret = calloc(num, size);
        if (ret == NULL) {
            printf("Memory Allocation Failed in File '%s' at Line %d!\n", FILE, LINE);
            perror("\tINFO");
            exit(1);
        }
        host_pointers.push_back(ret);
        return ret;
    }
#ifdef __CUDA_RUNTIME_H__
    void _CudaMalloc(void ** ptr, size_t size, const char * FILE, const int LINE) {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err != cudaSuccess) {
            printf("GPU Memory Allocation Failed in File '%s' at Line %d!\n", FILE, LINE);
            printf("\tINFO : %s", cudaGetErrorString(err));
            exit(1);
        }
        dev_pointers.push_back(* ptr);
    }
    void _CudaTimerStart(cudaStream_t stream, const char * FILE, const int LINE) {
        cudaError_t err = cudaEventRecord(start, stream);
        if (err != cudaSuccess) {
            printf("CUDA Timer Record Failed in File '%s' at Line %d!\n", FILE, LINE);
            printf("\tINFO : %s", cudaGetErrorString(err));
            exit(1);
        }
    }
    void _CudaTimerStop(cudaStream_t stream, const char * FILE, const int LINE) {
        cudaError_t err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess) {
            printf("CUDA Timer Record Failed in File '%s' at Line %d!\n", FILE, LINE);
            printf("\tINFO : %s", cudaGetErrorString(err));
            exit(1);
        }
    }
    float _CudaTimerWait(const char * FILE, const int LINE) {
        float t;
        cudaError_t err = cudaEventSynchronize(stop);
        if (err != cudaSuccess) {
            printf("CUDA Timer Synchronize Failed in File '%s' at Line %d!\n", FILE, LINE);
            printf("\tINFO : %s", cudaGetErrorString(err));
            exit(1);
        }
        err = cudaEventElapsedTime(&t, start, stop);
        if (err != cudaSuccess) {
            printf("CUDA Timer Failed in File '%s' at Line %d!\n", FILE, LINE);
            printf("\tINFO : %s", cudaGetErrorString(err));
            exit(1);
        }
        return t;
    }
    float _CudaTimerEnd(cudaStream_t stream, const char * FILE, const int LINE) {
        _CudaTimerStop(stream, FILE, LINE);
        return _CudaTimerWait(FILE, LINE);
    }
#endif  // #ifdef __CUDA_RUNTIME_H__
    ~auto_utility() {
        std::vector<void *>::iterator it;
        for (it = host_pointers.begin(); it != host_pointers.end(); it++)
            free(*it);
#ifdef __CUDA_RUNTIME_H__
        for (it = dev_pointers.begin(); it != dev_pointers.end(); it++)
            cudaFree(*it);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
#endif  // #ifdef __CUDA_RUNTIME_H__
    }
};

/* Utility Macros */
#define Auto_Utility() auto_utility au
#define Malloc(size) au._Malloc(size, __FILE__, __LINE__)
#define Calloc(num, size) au._Calloc(num, size, __FILE__, __LINE__)

#ifdef __CUDA_RUNTIME_H__

#define CudaSetDevice(dev) HANDLE_ERROR(cudaSetDevice(dev))

/* CUDA Memory function wrapper */
#define CudaMemset(devPtr, Byte, N) HANDLE_ERROR(cudaMemset(devPtr, Byte, N))
#define CudaMemcpy(destPtr, srcPtr, size, type) HANDLE_ERROR(cudaMemcpy(destPtr, srcPtr, size, type))
#define CudaMemcpyH2D(destPtr, srcPtr, size) HANDLE_ERROR(cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyHostToDevice))
#define CudaMemcpyD2H(destPtr, srcPtr, size) HANDLE_ERROR(cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToHost))
#define CudaMemcpyD2D(destPtr, srcPtr, size) HANDLE_ERROR(cudaMemcpy(destPtr, srcPtr, size, cudaMemcpyDeviceToDevice))
#define CudaFree(devPtr) HANDLE_ERROR(cudaFree(devPtr))

#define CudaMalloc(ptr, size) au._CudaMalloc((void**)&(ptr), size, __FILE__, __LINE__)

#define CudaBuffer(type, devPtr, num) \
            type * devPtr = NULL; CudaMalloc(devPtr, (num) * sizeof(type))
#define CudaBufferZero(type, devPtr, num) \
            CudaBuffer(type, devPtr, num); CudaMemset(devPtr, 0, (num) * sizeof(type))
#define CudaBufferCopy(type, devPtr, num, hostPtr) \
            CudaBuffer(type, devPtr, num); CudaMemcpyH2D(devPtr, hostPtr, (num) * sizeof(type))
#define CudaBufferFill(devPtr, size, hostPtr) \
            CudaMalloc(devPtr, size); CudaMemcpyH2D(devPtr, hostPtr, size)

/* CUDA timing wrapper */
#define CudaTimerStart() au._CudaTimerStart(0, __FILE__, __LINE__)
#define CudaTimerStop() au._CudaTimerStop(0, __FILE__, __LINE__)
#define CudaTimerWait() au._CudaTimerWait(__FILE__, __LINE__)
#define CudaTimerBegin() au._CudaTimerStart(0, __FILE__, __LINE__)
#define CudaTimerEnd() au._CudaTimerEnd(0, __FILE__, __LINE__)

/* CUDA Event function wrapper */
#define CudaEventCreate(event) HANDLE_ERROR(cudaEventCreate(&event))
#define CudaEventRecord(event, stream) HANDLE_ERROR(cudaEventRecord(event, stream))
#define CudaEventSynchronize(event) HANDLE_ERROR(cudaEventSynchronize(event))
#define CudaEventElapsedTime(time, begin, end) HANDLE_ERROR(cudaEventElapsedTime(&time, begin, end))
#define CudaEventDestroy(event) HANDLE_ERROR(cudaEventDestroy(event))

#endif  // #ifdef __CUDA_RUNTIME_H__

#endif  // #ifndef FROG_COMMON_H_INCLUDED
