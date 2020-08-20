/*-------------------------------------------------------------------------
 *
 * gpuqo.cuh
 *	  function prototypes and struct definitions for CUDA/Thrust code
 *
 * src/include/optimizer/gpuqo.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_CUH
#define GPUQO_CUH

#include <iostream>
#include "optimizer/gpuqo_common.h"
#include "optimizer/gpuqo_uninitalloc.cuh"

// I did not want to include the full c.h for fear of conflicts so I just 
// include the definitions (to get USE_ASSERT_CHECKING) and just define the
// Assert macro as in c.h
#include "pg_config.h"
#ifndef USE_ASSERT_CHECKING
#define Assert(condition)	((void)true)
#else
#include <assert.h>
#define Assert(p) assert(p)
#endif

// I do the same for the CHECK_FOR_INTERRUPTS macro
#include "signal.h"
extern "C" void ProcessInterrupts(void);
extern "C" volatile sig_atomic_t InterruptPending;

#define CHECK_FOR_INTERRUPTS() \
do { \
	if (InterruptPending) \
		ProcessInterrupts(); \
} while(0)

struct JoinRelation{
	RelationID id;
	RelationID left_relation_id;
	union{
		uint64_t left_relation_idx;
		struct JoinRelation* left_relation_ptr;
	};
	RelationID right_relation_id;
	union{
		uint64_t right_relation_idx;
		struct JoinRelation* right_relation_ptr;
	};
	double rows;
	double cost;
	EdgeMask edges;
	
#ifdef USE_ASSERT_CHECKING
	bool referenced;
#endif

public:
	__host__ __device__
	bool operator<(const JoinRelation &o) const
	{
		return cost < o.cost;
	}

	__host__ __device__
	bool operator>(const JoinRelation &o) const
	{
		return cost > o.cost;
	}

	__host__ __device__
	bool operator==(const JoinRelation &o) const
	{
		return cost == o.cost;
	}

	__host__ __device__
	bool operator<=(const JoinRelation &o) const
	{
		return cost <= o.cost;
	}

	__host__ __device__
	bool operator>=(const JoinRelation &o) const
	{
		return cost >= o.cost;
	}
};

extern std::ostream & operator<<(std::ostream &os, const JoinRelation& jr);

typedef thrust::device_vector<RelationID, uninitialized_allocator<RelationID> > uninit_device_vector_relid;
typedef thrust::device_vector<JoinRelation, uninitialized_allocator<JoinRelation> > uninit_device_vector_joinrel;

#endif							/* GPUQO_CUH */
