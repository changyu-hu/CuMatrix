#pragma once
#include "cuda_runtime.h"
#include <stdint.h>
#include <memory>
enum class CudaDataType : int32_t
{
    //! 32-bit floating point format.
    kFLOAT = 0,

    //! IEEE 16-bit floating-point format.
    kHALF = 1,

    //! 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4,

    //! 64-bit (on x64 system)
    kPOINTER = 5,

    //! variable size
    kSTRUCT = 6
};


//!
//! \brief  The GenericBuffer class is a templated class for buffers. Attributes to TensorRT samples.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
//! 
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    //! 
    GenericBuffer(size_t elementSize, CudaDataType type = CudaDataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mElementSize(elementSize)
        , mType(type)
        , mBuffer(nullptr)
        , mOwnership(true)
    {
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mElementSize(buf.getElementSize())
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
        , mOwnership(buf.getOwnerShip())
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = CudaDataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    // takeOwnership will be ignored if pPreAllocBuf is nullptr
    GenericBuffer(size_t size, CudaDataType type, size_t elementSize, void* pPreAllocBuf = nullptr, bool takeOwnership = false)
        : mSize(size)
        , mCapacity(size)
        , mElementSize(elementSize)
        , mType(type)
    {
        if (pPreAllocBuf == nullptr && size)
        {
            initializeWithSpace(size, type);
        }
        else
        {
            mOwnership = takeOwnership;
            mBuffer = pPreAllocBuf;

        }
    }

    ////!
    ////! \brief Construct a buffer with the specified allocation size in bytes.
    ////!
    //GenericBuffer(size_t size, CudaDataType type)
    //    : mSize(size)
    //    , mCapacity(size)
    //    , mType(type)
    //    , mOwnership(true)
    //{
    //    if (!allocFn(&mBuffer, this->nbBytes()))
    //    {
    //        throw std::bad_alloc();
    //    }
    //}

    void initializeWithSpace(size_t size, CudaDataType type) {
        mSize = size;
        mCapacity = size;
        mType = type;
        mOwnership = true;

        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeBuf();

            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    inline uint32_t getElementSize() const
    {
        
        return mElementSize;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return this->size() * getElementSize();
    }

    size_t nbBytes(size_t numElements) const
    {
        return numElements * getElementSize();
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeBuf();
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;

            mOwnership = true;
        }
    }

    ////!
    ////! \brief Overload of resize that accepts Dims
    ////!
    //void resize(const Dims& dims)
    //{
    //    return this->resize(volume(dims));
    //}

    void freeBuf() {
        if (mOwnership)
        {
            freeFn(mBuffer);
        }
    }

    ~GenericBuffer()
    {
        freeBuf();
    }

    bool getOwnerShip() {
        return mOwnership;
    }

protected:
    // size: number of elements, capacity: 
    size_t mSize{ 0 }, mCapacity{ 0 };
    size_t mElementSize{ 0 };
    CudaDataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
    bool mOwnership;

};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        auto retVal = cudaMalloc(ptr, size);
        CUDA_CHECK_RET(retVal);
        return retVal == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        CUDA_CHECK_RET(cudaFree(ptr));
    }
};

class ManagedAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        auto retVal = cudaMallocManaged(ptr, size);
        CUDA_CHECK_RET(retVal);
        return retVal == cudaSuccess;
    }
};

class ManagedFree
{
public:
    void operator()(void* ptr) const
    {
        CUDA_CHECK_RET(cudaFree(ptr));
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        CUDA_CHECK_RET(cudaMallocHost(ptr, size));
        //cudaHostAlloc
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        //free(ptr);
        CUDA_CHECK_RET(cudaFreeHost(ptr));
    }
};


using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

template <typename AllocFunc, typename FreeFunc, typename Class>
class ClassBuffer
{
public:
    using SharedPtr = std::shared_ptr<ClassBuffer>;
    typedef ClassBuffer* Ptr;

    ClassBuffer(bool callConstructor = false) {
        if (!allocFn(&data, sizeof(Class)))
        {
            throw std::bad_alloc();
        }

        if (callConstructor)
        {
            constructor();
        }

    }

    ~ClassBuffer()
    {
        freeFn(data);
    }

    void constructor() {
        new(data) Class();
    }

    Class* getData() {
        return (Class *) data;
    }

    Class& operator->() const
    {
        return *((Class *) data);
    }
protected:
    void* data;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

template<typename Class>
using ManagedClassBuffer = ClassBuffer<ManagedAllocator, ManagedFree, Class>;
template<typename Class>
class DeviceClassBuffer : public ClassBuffer<DeviceAllocator, DeviceFree, Class> 
{
public:

    void fromCPU(Class* pObj) {
        CUDA_CHECK_RET(cudaMemcpy(this->data, pObj, sizeof(Class), cudaMemcpyHostToDevice));
    }
};
