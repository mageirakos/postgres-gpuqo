/*------------------------------------------------------------------------
 *
 * gpuqo_idp.cu
 *      iterative dynamic programming implementation
 *
 * src/backend/optimizer/gpuqo/gpuqo_idp.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo.cuh"
#include "gpuqo_query_tree.cuh"

int gpuqo_idp_n_iters;

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp1_next(int gpuqo_algorithm, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_idp1(gpuqo_algorithm,new_info);

	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	delete new_info;
	freeQueryTree(new_qt);

	return new_qt_remap;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp1(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	info->n_iters = min(info->n_rels, gpuqo_idp_n_iters);

	LOG_PROFILE("IDP iteration with %d iterations (%d bits)\n", info->n_iters, BitmapsetN::SIZE);

	QueryTree<BitmapsetN> *qt = gpuqo_run_switch(gpuqo_algorithm, info);

	if (info->n_iters == info->n_rels){
		return qt;
	} else {
		list<remapper_transf_el_t<BitmapsetN> > remap_list;

		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = qt->id;
		list_el.to_idx = 0;
		list_el.qt = qt;
		remap_list.push_back(list_el);
		
		int j = 1;
		for (int i=0; i<info->n_rels; i++){
			if (!info->base_rels[i].id.isSubset(qt->id)){
				list_el.from_relid = info->base_rels[i].id;
				list_el.to_idx = j;
				list_el.qt = NULL;
				remap_list.push_back(list_el);

				j++;
			}
		}

		if (BitmapsetN::SIZE == 32 || remap_list.size() < 32) {
			return gpuqo_run_idp1_next<BitmapsetN, Bitmapset32>(
											gpuqo_algorithm, info, remap_list);
		} else if (BitmapsetN::SIZE == 64 || remap_list.size() < 64) {
			return gpuqo_run_idp1_next<BitmapsetN, Bitmapset64>(
											gpuqo_algorithm, info, remap_list);
		} else {
			printf("ERROR: too many relations\n");
			return NULL;	
		}
	}
}

template QueryTree<Bitmapset32> *gpuqo_run_idp1<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp1<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);