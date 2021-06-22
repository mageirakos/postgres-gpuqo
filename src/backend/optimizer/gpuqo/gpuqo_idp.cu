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

template<>
QueryTree<BitmapsetDynamic> *gpuqo_run_idp1(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetDynamic>* info)
{
	printf("CANNOT RUN IDP1 with Dynamic Bitmapset!\n");
	return NULL;
}

template QueryTree<Bitmapset32> *gpuqo_run_idp1<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp1<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic> *gpuqo_run_idp1<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*);


template<typename BitmapsetN>
QueryTree<BitmapsetN> *find_most_expensive_subtree(QueryTree<BitmapsetN> *qt, int max_size)
{
	Assert(qt != NULL && !qt->id.empty());
	Assert(max_size >= 1);

	if (qt->id.size() <= max_size) {
		return qt;
	} else {
		QueryTree<BitmapsetN> *lqt = find_most_expensive_subtree(qt->left, max_size);
		QueryTree<BitmapsetN> *rqt = find_most_expensive_subtree(qt->right, max_size);
		
		if (lqt->id.size() == 1)
			return rqt;
		else if (rqt->id.size() == 1)
			return lqt;
		else if (lqt->cost.total > rqt->cost.total)
			return lqt;
		else if (lqt->cost.total < rqt->cost.total)
			return rqt;
		else if (lqt->id.size() < rqt->id.size())
			return lqt;
		else
			return rqt;
	}
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

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp2_rec(int gpuqo_algorithm, 
						QueryTree<BitmapsetOuter> *goo_qt,
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);
	QueryTree<BitmapsetInner> *new_goo_qt =remapper.remapQueryTreeFwd(goo_qt);

	new_info->n_iters = min(new_info->n_rels, gpuqo_idp_n_iters);

	BitmapsetInner reopTables = find_most_expensive_subtree(new_goo_qt, new_info->n_iters)->id;

	list<remapper_transf_el_t<BitmapsetInner> > reopt_remap_list;
	int i = 0;
	while (!reopTables.empty()) {
		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopTables.lowest();
		list_el.to_idx = i++;
		list_el.qt = NULL;
		reopt_remap_list.push_back(list_el);

		reopTables -= list_el.from_relid;
	}

	QueryTree<BitmapsetInner> *reopt_qt;

	if (BitmapsetInner::SIZE == 32 || reopt_remap_list.size() < 32) {
		reopt_qt = gpuqo_run_idp2_dp<BitmapsetInner, Bitmapset32>(
								gpuqo_algorithm, new_info, reopt_remap_list);
	} else if (BitmapsetInner::SIZE == 64 || reopt_remap_list.size() < 64) {
		reopt_qt = gpuqo_run_idp2_dp<BitmapsetInner, Bitmapset64>(
								gpuqo_algorithm, new_info, reopt_remap_list);
	} else {
		printf("ERROR: too many relations\n");
		return NULL;	
	}

	QueryTree<BitmapsetInner> *res_qt;
	if (new_info->n_iters == new_info->n_rels){
		res_qt = reopt_qt;
	} else {
		list<remapper_transf_el_t<BitmapsetInner> > next_remap_list;

		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopt_qt->id;
		list_el.to_idx = 0;
		list_el.qt = reopt_qt;
		next_remap_list.push_back(list_el);
		
		int j = 1;
		for (int i=0; i<new_info->n_rels; i++){
			if (!new_info->base_rels[i].id.isSubset(reopt_qt->id)){
				list_el.from_relid = new_info->base_rels[i].id;
				list_el.to_idx = j++;
				list_el.qt = NULL;
				next_remap_list.push_back(list_el);
			}
		}

		if (BitmapsetInner::SIZE == 32 || next_remap_list.size() < 32) {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, Bitmapset32>(
					gpuqo_algorithm, new_goo_qt, new_info, next_remap_list);
		} else if (BitmapsetInner::SIZE == 64 || next_remap_list.size() < 64) {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, Bitmapset64>(
					gpuqo_algorithm, new_goo_qt, new_info, next_remap_list);
		} else {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, BitmapsetDynamic>(
					gpuqo_algorithm, new_goo_qt, new_info, next_remap_list);
		}
	}

	QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(res_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_goo_qt);
	freeQueryTree(res_qt);

	return out_qt;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp2(int gpuqo_algorithm, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	QueryTree<BitmapsetN> *goo_qt = gpuqo_cpu_goo(info);
	list<remapper_transf_el_t<BitmapsetN> > remap_list;
	for (int i=0; i<info->n_rels; i++){
		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = info->base_rels[i].id;
		list_el.to_idx = i;
		list_el.qt = NULL;
		remap_list.push_back(list_el);
	}

	QueryTree<BitmapsetN> *out_qt = gpuqo_run_idp2_rec<BitmapsetN,BitmapsetN>(gpuqo_algorithm, goo_qt, info, remap_list);

	freeQueryTree(goo_qt);

	return out_qt;
}

template QueryTree<Bitmapset32> *gpuqo_run_idp2<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64> *gpuqo_run_idp2<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic> *gpuqo_run_idp2<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*);