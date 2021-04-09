/*-------------------------------------------------------------------------
 *
 * gpuqo_planner_info.cuh
 *	  structure for planner info in GPU memory
 *
 * src/include/optimizer/gpuqo.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_PLANNER_INFO_CUH
#define GPUQO_PLANNER_INFO_CUH

#include <optimizer/gpuqo_common.h>

namespace gpuqo_c{
	#include <optimizer/gpuqo_planner_info.h>
};

typedef Bitmapset32 EdgeMask;
typedef Bitmapset32 RelationID;

typedef struct QueryTree{
	RelationID id;
	float rows;
	float cost;
	struct QueryTree* left;
	struct QueryTree* right;
} QueryTree;


struct BaseRelation{
	RelationID id;
	float rows;
	float tuples;
	int off_fk_selecs;
	int n_fk_selecs;
};

struct GpuqoPlannerInfo{
	unsigned int size;
	int n_rels;
	BaseRelation base_rels[32];
	EdgeMask edge_table[32];
	EdgeMask indexed_edge_table[32];
	RelationID subtrees[32];
	int n_fk_selecs;
	unsigned int* fk_selec_idxs;
	float* fk_selec_sels;
	int n_eq_classes;
	int n_eq_class_sels;
	RelationID* eq_classes;
	float* eq_class_sels;
};

GpuqoPlannerInfo* convertGpuqoPlannerInfo(gpuqo_c::GpuqoPlannerInfo *info_c);
GpuqoPlannerInfo* copyToDeviceGpuqoPlannerInfo(GpuqoPlannerInfo *info);
GpuqoPlannerInfo* deleteGpuqoPlannerInfo(GpuqoPlannerInfo *info);

#endif							/* GPUQO_PLANNER_INFO_CUH */
