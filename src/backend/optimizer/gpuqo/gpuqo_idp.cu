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

int gpuqo_idp_n_iters;

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	info->n_iters = min(info->n_rels, gpuqo_idp_n_iters);

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

		Remapper<BitmapsetN> remapper(remap_list);
		GpuqoPlannerInfo<BitmapsetN> *new_info =remapper.remapPlannerInfo(info);

		QueryTree<BitmapsetN> *new_qt = gpuqo_run_idp(gpuqo_algorithm,new_info);

		delete new_info;

		remapper.remapQueryTree(new_qt);

		return new_qt;
	}
}

template QueryTree<Bitmapset32> *gpuqo_run_idp<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);