/*-------------------------------------------------------------------------
 *
 * gpuqo_uninitalloc.cuh
 *	  definition of uninitialized_vector type, which is used in device_vector
 *    to prevent initialization to 0 by default thus preventing some overhead.
 *
 * src/include/optimizer/gpuqo_uninitalloc.cuh
 *
 *-------------------------------------------------------------------------
 */

/****** Taken from thrust samples *****/
// Occasionally, it is advantageous to avoid initializing the individual
// elements of a device_vector. For example, the default behavior of
// zero-initializing numeric data may introduce undesirable overhead.
// This example demonstrates how to avoid default construction of a
// device_vector's data by using a custom allocator.

#ifndef GPUQO_UNINITALLOC_CUH
#define GPUQO_UNINITALLOC_CUH

#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>

// uninitialized_allocator is an allocator which
// derives from device_allocator and which has a
// no-op construct member function
template<typename T>
  struct uninitialized_allocator
    : thrust::device_allocator<T>
{
  // the default generated constructors and destructors are implicitly
  // marked __host__ __device__, but the current Thrust device_allocator
  // can only be constructed and destroyed on the host; therefore, we
  // define these as host only
  __host__
  uninitialized_allocator() {}
  __host__
  uninitialized_allocator(const uninitialized_allocator & other)
    : thrust::device_allocator<T>(other) {}
  __host__
  ~uninitialized_allocator() {}

#if THRUST_CPP_DIALECT >= 2011
  uninitialized_allocator & operator=(const uninitialized_allocator &) = default;
#endif

  // for correctness, you should also redefine rebind when you inherit
  // from an allocator type; this way, if the allocator is rebound somewhere,
  // it's going to be rebound to the correct type - and not to its base
  // type for U
  template<typename U>
  struct rebind
  {
    typedef uninitialized_allocator<U> other;
  };

  // note that construct is annotated as
  // a __host__ __device__ function
  __host__ __device__
  void construct(T *)
  {
    // no-op
  }
};

#endif							/* GPUQO_UNINITALLOC_CUH */
