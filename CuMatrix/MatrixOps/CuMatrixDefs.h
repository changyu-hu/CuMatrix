#pragma once
#include "cuda_runtime.h"
#include <iostream>

#define GPU_CPU_INLINE_FUNC  __forceinline__ __device__ __host__
#define GPU_CPU_FUNC_NO_INLINE  __device__ __host__

#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

#ifdef __CUDACC__
#define HOST_FUNC               __host__
#define DEVICE_FUNC             __device__
#define HOST_INLINE_FUNC        __host__ __forceinline__
#define DEVICE_INLINE_FUNC      __device__ __forceinline__
#define HOST_DEVICE_FUNC        __host__ __device__
#define HOST_DEVICE_INLINE_FUNC __host__ __device__ __forceinline__
#else
#define HOST_FUNC
#define DEVICE_FUNC
#define HOST_INLINE_FUNC
#define DEVICE_INLINE_FUNC
#define HOST_DEVICE_FUNC
#define HOST_DEVICE_INLINE_FUNC
#endif


// to avoid the triple bracket when calling global function
// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define CUDA_CHECK_RET(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << " - " << cudaGetErrorString(ret) <<  std::endl;                                                 \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDA_ERROR(val) Check((val), #val, __FILE__, __LINE__)
template <typename T>
void Check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CMP_EPSILON 0.00001f
#define CMP_EPSILON2 (CMP_EPSILON * CMP_EPSILON)
#define IS_ZERO_APPROX(x) (fabs(x) < CMP_EPSILON)