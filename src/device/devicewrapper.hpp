/**
 * PPAM CUDA TUTORIAL
 *
 * Helper class encapsulating everyday CUDA API operations:
 * - device info
 * - error-to-human-readable-string mapping
 * - high-resolution timer (only for stream 0!)
 *
 * (c) 2012,2013 dominik.goeddeke@math.tu-dortmund.de
 */

#ifndef DEVICEWRAPPER_GUARD
#define DEVICEWRAPPER_GUARD

// standard includes
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// CUDA include, only necessary if not compiling with nvcc
#include <cuda_runtime.h>



class DeviceWrapper
{
  private:
    /**
     * device that this wrapper is bound to, set via constructor
     */
    int _device;
    
    /**
     * number of devices in the system
     */
    int _numDevices;
    
    /**
     * events for the built-in timer
     */
    cudaEvent_t _start, _stop;


  public:
    /**
     * Constructor: binds this wrapper to device 0
     */
    DeviceWrapper ()
    {
      std::cout << "Initialising Dominik's simplistic device wrapper for device 0" << std::endl;
      cudaError_t error = cudaGetDeviceCount (&_numDevices);
      if (error != cudaSuccess)
      {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "No CUDA-capable devices found, exiting." << std::endl;
        exit(1);
      }
      if (_numDevices == 0)
      {
        std::cerr << "No CUDA-capable devices found, exiting." << std::endl;
        exit(1);
      }

      _device = 0;
      
      cudaEventCreate (&_start);
      cudaEventCreate (&_stop);
    }
    
    /**
     * Constructor: binds this wrapper to a given device
     */
    DeviceWrapper (const int device)
    {
      std::cout << "Initialising Dominik's simplistic device wrapper for device " << device << std::endl;
      cudaError_t error = cudaGetDeviceCount (&_numDevices);
      if (error != cudaSuccess)
      {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(error) << std::endl;
        std::cerr << "No CUDA-capable devices found, exiting." << std::endl;
        exit(1);
      }
      if (_numDevices == 0)
      {
        std::cerr << "No CUDA-capable devices found, exiting." << std::endl;
        exit(1);
      }
 
      if (device >= 0 && device < _numDevices)
      {
        _device = device;
      }
      else
      {
        std::cerr << "Invalid device number, using device 0" << std::endl;
        _device = 0;
      }
      
      error = cudaEventCreate (&_start);
      if (error != cudaSuccess)
      {
        std::cerr << "CUDA ERROR at cudaEventCreate(): " << cudaGetErrorString(error) << std::endl;
        exit(1);
      }
      error = cudaEventCreate (&_stop);
      if (error != cudaSuccess)
      {
        std::cerr << "CUDA ERROR at cudaEventCreate(): " << cudaGetErrorString(error) << std::endl;
        exit(1);
      }
    }
    
    /**
     * Destructor
     */
    ~DeviceWrapper ()
    {
      cudaEventDestroy (_start);
      cudaEventDestroy (_stop);
    }
    
    /**
     * Returns the number of the device for use in calls to cudaSetDevice()
     */
    int getDeviceNumber()
    {
      return _device;
    }
    
    /**
     * processes given return code, exits on error (after translating
     * into human-readable form)
     */
    void processReturnCode (const cudaError_t returnCode, const char* label)
    {
      #ifdef DEBUG
        if (returnCode != cudaSuccess)
        {
          std::cerr << "CUDA ERROR (at " << label << "): " << cudaGetErrorString (returnCode) << std::endl;
          std::cerr << "Exiting..." << std::endl;
          exit(1);
        }
      #endif
    }
    
    /**
     * prints out brief information about this device
     */
    void printDeviceName ()
    {
      cudaDeviceProp prop;
      cudaError_t err = cudaGetDeviceProperties (&prop, _device);
      processReturnCode(err, "DeviceWrapper::printDeviceName()");
      std::cout << "Device " << _device << ": " << prop.name << std::endl;
    }
    
    /**
     * prints out lots of information about this device
     */
    void printDeviceProperties ()
    {
      cudaDeviceProp prop;
      cudaError_t err = cudaGetDeviceProperties (&prop, _device);
      processReturnCode(err, "DeviceWrapper::printDeviceProperties()");
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "Device properties of device " << _device << ": " << prop.name << std::endl;
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
      std::cout << "Number of cores   : " << prop.multiProcessorCount << std::endl;
      std::cout << "Clock rate        : " << std::fixed<<std::setprecision(2) << prop.clockRate/1e6 << " GHz" << std::endl;
      std::cout << "Total memory      : " << std::fixed<<std::setprecision(2) << prop.totalGlobalMem/pow(2.0,30) << " GB" << std::endl;
      std::cout << "Memory bus width  : " << prop.memoryBusWidth << " bit" << std::endl;
      std::cout << "Memory clock rate : " << std::fixed<<std::setprecision(2) << prop.memoryClockRate/1e6 << " GHz" << std::endl;
      bool b = prop.ECCEnabled>0 ? true : false;
      std::cout << "ECC memory        : " << std::boolalpha << b << std::endl;
      std::cout << "L2 cache size     : " << std::fixed<<std::setprecision(0) << prop.l2CacheSize/1024.0 << " kB" << std::endl;
      std::cout << "Async. engines    : " << prop.asyncEngineCount << std::endl;
      b = prop.concurrentKernels>0 ? true : false;
      std::cout << "Concurrent kernels: " << std::boolalpha << b << std::endl;
      std::cout << "Constant memory   : " << std::fixed<<std::setprecision(0) << prop.totalConstMem/1024.0 << " kB" << std::endl;
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "Warp size                     : " << prop.warpSize << std::endl;
      std::cout << "Max threads per core          : " << prop.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "Max threads per block         : " << prop.maxThreadsPerBlock << std::endl;
      std::cout << "Max block dimensions          : " << prop.maxThreadsDim[0] << "x" << prop.maxThreadsDim[1] << "x"<< prop.maxThreadsDim[2] << std::endl;
      std::cout << "Max grid dimensions           : " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x"<< prop.maxGridSize[2] << std::endl;
      std::cout << "Max 32bit registers per block : " << prop.regsPerBlock << std::endl;
      std::cout << "Max shared memory per block   : " << std::fixed<<std::setprecision(0) << prop.sharedMemPerBlock/1024.0 << " kB" << std::endl;
      std::cout << "-------------------------------------------------" << std::endl;
    }
    
    /**
     * starts device timer, only works for stream 0
     * resolution: 0.5 microseconds
     */
    void startDeviceTimer ()
    {
      #ifdef DEBUG
        std::cout << "Warning: This timer assumes everything happens in stream 0" << std::endl;
      #endif
      cudaEventRecord (_start, 0);
    }

    /**
     * stop device timer and return elapsed time in milliseconds
     */
    float stopDeviceTimer ()
    {
      cudaEventRecord (_stop, 0);
      cudaEventSynchronize (_stop);
      float elapsed;
      cudaEventElapsedTime (&elapsed, _start, _stop);
      return elapsed;
    }
    
};

#endif // DEVICEWRAPPER_GUARD
