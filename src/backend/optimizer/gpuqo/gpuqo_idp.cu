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
int gpuqo_idp_type;

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp1_next(int gpuqo_algorithm, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_idp1(gpuqo_algorithm,new_info);

	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);

	return new_qt_remap;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp1(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	info->n_iters = min(info->n_rels, gpuqo_idp_n_iters);

	LOG_PROFILE("IDP1 iteration with %d iterations: %d sets remaining (%d bits)\n", info->n_iters, info->n_rels, BitmapsetN::SIZE);

	QueryTree<BitmapsetN> *qt = gpuqo_run_switch(gpuqo_algorithm, info);

	if (info->n_iters == info->n_rels){
		return qt;
	}

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
			list_el.to_idx = j++;
			list_el.qt = NULL;
			remap_list.push_back(list_el);
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

template QueryTree<Bitmapset32> *gpuqo_run_idp1<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp1<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp2_next(int gpuqo_algorithm, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_idp2(gpuqo_algorithm,new_info);

	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);

	return new_qt_remap;
}

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp2_dp(int gpuqo_algorithm, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);
	new_info->n_iters = new_info->n_rels;

	LOG_PROFILE("IDP2 iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_switch(gpuqo_algorithm,new_info);

	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);

	return new_qt_remap;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp2(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	info->n_iters = min(info->n_rels, gpuqo_idp_n_iters);

	LOG_PROFILE("IDP2 iteration with %d iterations: %d sets remaining (%d bits)\n", info->n_iters, info->n_rels, BitmapsetN::SIZE);

	BitmapsetN reopTables;
	if (info->n_iters == info->n_rels) {
		reopTables = BitmapsetN(0);
		for (int i = 0; i < info->n_rels; i++) {
			reopTables |= info->base_rels[i].id;
		}
	} else {
		QueryTree<BitmapsetN> *goo_qt = gpuqo_cpu_goo(info);

		Assert(goo_qt->left != NULL && goo_qt->right != NULL);
		
		if (goo_qt->left != NULL 
				&& (goo_qt->right != NULL 
					|| goo_qt->left->id.size() < goo_qt->right->id.size()))
		{
			reopTables = goo_qt->left->id;
		} else if (goo_qt->right != NULL) {
			reopTables = goo_qt->right->id;
		} else {
			printf("FATAL ERROR\n");
			abort();
		}
		freeQueryTree(goo_qt);
	}

	list<remapper_transf_el_t<BitmapsetN> > remap_list;
	int i = 0;
	while (!reopTables.empty()) {
		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = reopTables.lowest();
		list_el.to_idx = i++;
		list_el.qt = NULL;
		remap_list.push_back(list_el);

		reopTables -= list_el.from_relid;
	}

	QueryTree<BitmapsetN> *qt;

	if (BitmapsetN::SIZE == 32 || remap_list.size() < 32) {
		qt = gpuqo_run_idp2_dp<BitmapsetN, Bitmapset32>(
										gpuqo_algorithm, info, remap_list);
	} else if (BitmapsetN::SIZE == 64 || remap_list.size() < 64) {
		qt = gpuqo_run_idp2_dp<BitmapsetN, Bitmapset64>(
										gpuqo_algorithm, info, remap_list);
	} else {
		printf("ERROR: too many relations\n");
		return NULL;	
	}

	if (info->n_iters == info->n_rels){
		return qt;
	}

	remap_list.clear();

	remapper_transf_el_t<BitmapsetN> list_el;
	list_el.from_relid = qt->id;
	list_el.to_idx = 0;
	list_el.qt = qt;
	remap_list.push_back(list_el);
	
	int j = 1;
	for (int i=0; i<info->n_rels; i++){
		if (!info->base_rels[i].id.isSubset(qt->id)){
			list_el.from_relid = info->base_rels[i].id;
			list_el.to_idx = j++;
			list_el.qt = NULL;
			remap_list.push_back(list_el);
		}
	}

	if (BitmapsetN::SIZE == 32 || remap_list.size() < 32) {
		return gpuqo_run_idp2_next<BitmapsetN, Bitmapset32>(
										gpuqo_algorithm, info, remap_list);
	} else if (BitmapsetN::SIZE == 64 || remap_list.size() < 64) {
		return gpuqo_run_idp2_next<BitmapsetN, Bitmapset64>(
										gpuqo_algorithm, info, remap_list);
	} else {
		return gpuqo_run_idp2_next<BitmapsetN, BitmapsetDynamic>(
										gpuqo_algorithm, info, remap_list);
	}
}

template QueryTree<Bitmapset32> *gpuqo_run_idp2<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp2<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic> *gpuqo_run_idp2<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*);