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

 extern "C" QueryTree *gpuqo_run(int gpuqo_algorithm, gpuqo_c::GpuqoPlannerInfo* info_c)
 {
	GpuqoPlannerInfo *info = convertGpuqoPlannerInfo(info_c);

	if (gpuqo_spanning_tree_enable){
        minimumSpanningTree(info);
        buildSubTrees(info->subtrees, info);
    }

    int* remap_table_fw = new int[info->n_rels];
    int* remap_table_bw = new int[info->n_rels];

    makeBFSIndexRemapTables(remap_table_fw, remap_table_bw, info);
    remapPlannerInfo(info, remap_table_fw);

	QueryTree* query_tree;
	switch (gpuqo_algorithm)
	{
	case GPUQO_DPSIZE:
		query_tree = gpuqo_dpsize(info);
		break;
	case GPUQO_DPSUB:
		query_tree = gpuqo_dpsub(info);
		break;
	case GPUQO_CPU_DPSIZE:
		query_tree = gpuqo_cpu_dpsize(info);
		break;
	case GPUQO_CPU_DPSUB:
		query_tree = gpuqo_cpu_dpsub(info);
		break;
	case GPUQO_CPU_DPCCP:
		query_tree = gpuqo_cpu_dpccp(info);
		break;
	case GPUQO_DPE_DPSIZE:
		query_tree = gpuqo_dpe_dpsize(info);
		break;
	case GPUQO_DPE_DPSUB:
		query_tree = gpuqo_dpe_dpsub(info);
		break;
	case GPUQO_DPE_DPCCP:
		query_tree = gpuqo_dpe_dpccp(info);
		break;
	 default: 
		// impossible branch but without it the compiler complains
		query_tree = NULL;
		break;
	 
	
	}

	remapQueryTree(query_tree, remap_table_bw);

    delete remap_table_fw;
    delete remap_table_bw;
 
	free(info);
 
	return query_tree;
 }