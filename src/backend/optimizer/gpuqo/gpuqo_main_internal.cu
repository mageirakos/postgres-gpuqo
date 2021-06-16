/*------------------------------------------------------------------------
 *
 * gpuqo_main_internal.cu
 *      implementation of run function
 *
 * src/backend/optimizer/gpuqo/gpuqo_main_internal.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo.cuh"
#include "gpuqo_query_tree.cuh"

#ifdef GPUQO_PRINT_N_JOINS
__device__ unsigned long long join_counter;
#endif

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_switch(int gpuqo_algorithm, 
										GpuqoPlannerInfo<BitmapsetN>* info)
{
	switch (gpuqo_algorithm)
	{
	case GPUQO_DPSIZE:
		return gpuqo_dpsize(info);
		break;
	case GPUQO_DPSUB:
		return gpuqo_dpsub(info);
		break;
	case GPUQO_CPU_DPSIZE:
		return gpuqo_cpu_dpsize(info);
		break;
	case GPUQO_CPU_DPSUB:
		return gpuqo_cpu_dpsub(info);
		break;
	case GPUQO_CPU_DPSUB_PARALLEL:
		return gpuqo_cpu_dpsub_parallel(info);
		break;
	case GPUQO_CPU_DPSUB_BICC:
		return gpuqo_cpu_dpsub_bicc(info);
		break;
	case GPUQO_CPU_DPSUB_BICC_PARALLEL:
		return gpuqo_cpu_dpsub_bicc_parallel(info);
		break;
	case GPUQO_CPU_DPCCP:
		return gpuqo_cpu_dpccp(info);
		break;
	case GPUQO_DPE_DPSIZE:
		return gpuqo_dpe_dpsize(info);
		break;
	case GPUQO_DPE_DPSUB:
		return gpuqo_dpe_dpsub(info);
		break;
	case GPUQO_DPE_DPCCP:
		return gpuqo_dpe_dpccp(info);
		break;
	case GPUQO_CPU_GOO:
		return gpuqo_cpu_goo(info);
		break;
	case GPUQO_CPU_IKKBZ:
		return gpuqo_cpu_ikkbz(info);
		break;
	case GPUQO_CPU_LINEARIZED_DP:
		return gpuqo_cpu_linearized_dp(info);
		break;
	 default: 
		// impossible branch but without it the compiler complains
		return NULL;
		break;
	}
}

template<>
QueryTree<BitmapsetDynamic> *gpuqo_run_switch(int gpuqo_algorithm, 
										GpuqoPlannerInfo<BitmapsetDynamic>* info)
{
	switch (gpuqo_algorithm)
	{
	case GPUQO_CPU_GOO:
		return gpuqo_cpu_goo(info);
		break;
	case GPUQO_CPU_IKKBZ:
		return gpuqo_cpu_ikkbz(info);
		break;
	case GPUQO_CPU_LINEARIZED_DP:
		return gpuqo_cpu_linearized_dp(info);
		break;
	 default: 
		// impossible branch but without it the compiler complains
		return NULL;
		break;
	}
}

template<typename BitmapsetN>
static QueryTreeC *__gpuqo_run(int gpuqo_algorithm, GpuqoPlannerInfoC* info_c)
{
	GpuqoPlannerInfo<BitmapsetN> *info = convertGpuqoPlannerInfo<BitmapsetN>(info_c);

	if (gpuqo_spanning_tree_enable){
		minimumSpanningTree(info);
		buildSubTrees(info->subtrees, info);
	}

	Remapper<BitmapsetN,BitmapsetN> remapper = makeBFSIndexRemapper(info);
	GpuqoPlannerInfo<BitmapsetN> *remap_info = remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetN> *query_tree;
	if (gpuqo_idp_n_iters <= 1 || gpuqo_idp_n_iters >= info->n_rels)
		query_tree = gpuqo_run_switch(gpuqo_algorithm, remap_info);
	else{
		switch(gpuqo_idp_type) {
			case GPUQO_IDP1:
				query_tree = gpuqo_run_idp1(gpuqo_algorithm, remap_info);
				break;
			case GPUQO_IDP2:
				query_tree = gpuqo_run_idp2(gpuqo_algorithm, remap_info);
				break;
			default:
				printf("Unkonwn IDP type\n");
		}
	}

	QueryTree<BitmapsetN> *new_qt = remapper.remapQueryTree(query_tree);
	freeQueryTree(query_tree);

	freeGpuqoPlannerInfo(remap_info);
	freeGpuqoPlannerInfo(info);

	return convertQueryTree(new_qt);
}

template<typename BitmapsetN>
static QueryTreeC *__gpuqo_run_idp2(int gpuqo_algorithm, GpuqoPlannerInfoC* info_c)
{
	GpuqoPlannerInfo<BitmapsetN> *info = convertGpuqoPlannerInfo<BitmapsetN>(info_c);

	if (gpuqo_spanning_tree_enable){
		GpuqoPlannerInfo<BitmapsetN> *new_info = minimumSpanningTree(info);
		freeGpuqoPlannerInfo(info);
		info = new_info;

		buildSubTrees(info->subtrees, info);
	}

	Remapper<BitmapsetN,BitmapsetN> remapper = makeBFSIndexRemapper(info);
	GpuqoPlannerInfo<BitmapsetN> *remap_info = remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetN> *query_tree = gpuqo_run_idp2(gpuqo_algorithm, remap_info);


	QueryTree<BitmapsetN> *new_qt = remapper.remapQueryTree(query_tree);
	freeQueryTree(query_tree);

	freeGpuqoPlannerInfo(remap_info);
	freeGpuqoPlannerInfo(info);

	return convertQueryTree(new_qt);
}

extern "C" QueryTreeC *gpuqo_run(int gpuqo_algorithm, GpuqoPlannerInfoC* info_c){
	if (info_c->n_rels < 32){
		return __gpuqo_run<Bitmapset32>(gpuqo_algorithm, info_c);
	} else if (info_c->n_rels < 64){
		return __gpuqo_run<Bitmapset64>(gpuqo_algorithm, info_c);
	} else if (gpuqo_algorithm == GPUQO_CPU_GOO
			|| gpuqo_algorithm == GPUQO_CPU_IKKBZ
			|| gpuqo_algorithm == GPUQO_CPU_LINEARIZED_DP
	){
		return __gpuqo_run<BitmapsetDynamic>(gpuqo_algorithm, info_c);
	} else {
		if (gpuqo_idp_n_iters > 1 && gpuqo_idp_n_iters < 64 
			&& gpuqo_idp_type == GPUQO_IDP2)
		{
			return __gpuqo_run_idp2<BitmapsetDynamic>(gpuqo_algorithm, info_c);
		} else {
			printf("ERROR: too many relations. Use IDP2.\n");
			return NULL;	
		}
	}
}
