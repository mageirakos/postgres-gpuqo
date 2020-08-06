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

struct JoinRelation{
	unsigned int left_relation_idx;
	unsigned int right_relation_idx;
	unsigned int rows;
	double cost;
	// I could store more information but I'm striving to keep it as small as 
	// possible

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

std::ostream & operator<<(std::ostream &os, const JoinRelation& jr)
{
	os<<"("<<jr.left_relation_idx<<","<<jr.right_relation_idx;
	os<<"): rows="<<jr.rows<<", cost="<<jr.cost;
	return os;
}

#endif							/* GPUQO_CUH */
