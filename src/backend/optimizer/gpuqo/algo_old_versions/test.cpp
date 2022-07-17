#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
struct Foo {
    T x;
    int y;
};

// Functor
template<typename T>
struct SortFoo {
    // reference (&)
    bool operator()(const Foo<T>& lhs, const Foo<T>& rhs) {
        return lhs.x < rhs.x;
    }
};

int main() {
    std::vector<Foo<int>> foos;
    
    foos.push_back({1, 2});
    foos.push_back({-1, 2});
    foos.push_back({10, 2});
    foos.push_back({23, 2});
    foos.push_back({2, 2});
    foos.push_back({5, 2});
    foos.push_back({7, 2});
    foos.push_back({9, 2});

    std::sort(foos.begin(), foos.end(), SortFoo<int>()); // begin/end return iterators

    for (const auto& foo : foos) {
        std::cout << foo.x << " " << foo.y << std::endl;
    }
    
    return 0;
}


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

// int gpuqo_idp_n_iters;
// int gpuqo_idp_type;

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp1_next(int gpuqo_algo, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info =remapper.remapPlannerInfo(info);

	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_idp1(gpuqo_algo,new_info);

	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);

	return new_qt_remap;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp1(int gpuqo_algo, 
									GpuqoPlannerInfo<BitmapsetN>* info)
{
	info->n_iters = min(info->n_rels, gpuqo_idp_n_iters);

	LOG_PROFILE("IDP1 iteration with %d iterations: %d sets remaining (%d bits)\n", info->n_iters, info->n_rels, BitmapsetN::SIZE);

	QueryTree<BitmapsetN> *qt = gpuqo_run_switch(gpuqo_algo, info);

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
										gpuqo_algo, info, remap_list);
	} else if (BitmapsetN::SIZE == 64 || remap_list.size() < 64) {
		return gpuqo_run_idp1_next<BitmapsetN, Bitmapset64>(
										gpuqo_algo, info, remap_list);
	} else {
		return gpuqo_run_idp1_next<BitmapsetN, BitmapsetDynamic>(
										gpuqo_algo, info, remap_list);
	}
}

template<>
QueryTree<BitmapsetDynamic> *gpuqo_run_idp1(int gpuqo_algo, 
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
QueryTree<BitmapsetOuter> *gpuqo_run_idp2_dp(int gpuqo_algo, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	new_info->n_iters = new_info->n_rels;

	LOG_PROFILE("IDP2 iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	// edw trexei o adistoixos dp algorithmos (mpdp emeis)
	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_switch(gpuqo_algo, new_info);
	// kai mas epistrefei pointer sto neo query tree pou einai optimized (*new_qt)
	// to opoio pame kai to kanoume remapp me vasi to remap_list pou mas dwthike
	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	// den xreiazomaste pleon to neq_qt afou to kaname remap ara ta eleutherwnoume
	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);
	
	// epistrefei to remapped optimized query Tree T'
	return new_qt_remap;
}


// se autin tin sunartisi exoume idi trexei to heuristic GOO 
// praktika apo pseudokwdika einai line 2-9
template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_idp2_rec(int gpuqo_algo, 
					QueryTree<BitmapsetOuter> *goo_qt,
					GpuqoPlannerInfo<BitmapsetOuter>* info,
					list<remapper_transf_el_t<BitmapsetOuter> > &remap_list,
					int n_iters) 
{
	// // -----LINE 2 THE WHILE LOOP IS *THIS* FUNCTION CALLED RECURSIVELY 
	// sto prwto iteration den tha kanei tipota to remap, giati einai san remap ston eauto tou
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	// ? read under gpuqo_remapper.cu 
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	QueryTree<BitmapsetInner> *new_goo_qt = remapper.remapQueryTreeFwd(goo_qt);

	// an exoun meinei ligotera relations sto T apo o,ti n_iters(25 vazw egw sunithws) tote krata auto san n_iters
	new_info->n_iters = min(new_info->n_rels, n_iters);
	// vriksei to T', pou einai to pio akrivo dedro EWS KAI n_iters relations
	// epistrefei to dedro T' pou einai Bitmapset (reopTables)
	// -----LINE 3 PICK T' SUBTREE OF T SO THAT COST IS MAXIMAL
	// to reopTables edw einai einai adigrafo tou maximal->id, ara allages sto reopTables den allazei to maximal->id
	BitmapsetInner reopTables = find_most_expensive_subtree(new_goo_qt, new_info->n_iters)->id;

	// dimiourgei pali mia lista me ta remaps tou reopt pleon (to T')
	list<remapper_transf_el_t<BitmapsetInner> > reopt_remap_list;
	int i = 0;
	// Auto nomizw einai ena prwto remapping tou subtree T' se BFS numbering wste na borei na bei sto dp kai na tre3ei swsta kai na to kanei optimize
	while (!reopTables.empty()) {
		remapper_transf_el_t<BitmapsetInner> list_el;
		// ? apo opoio relation einai to lowest (psa3e na katalaveis kalutera kai zwgrafise to)
		list_el.from_relid = reopTables.lowest();
		// ? paei kai au3anei kata 1 
		list_el.to_idx = i++;
		list_el.qt = NULL;
		// to prosthetei stin lista tou remap
		reopt_remap_list.push_back(list_el);
		// to vgazei apo to reoptable (pws ginetai auti i pra3i thelw na katalavw, einai san to unset?)
		reopTables -= list_el.from_relid;
	}

	// ara pleon ena pointer sto query tree T' dimiourgeite
	QueryTree<BitmapsetInner> *reopt_qt;

	// kalei ton MPDP i opoion dp algorithmo xrisimopoioume telos padwn
	// me tropo wste pada na benei to mikrotero Bitmapset Size
	// -----LINE 5 OPTIMIZE JOIN ORDER OF RELATIONS IN T' USING DP
	if (BitmapsetInner::SIZE == 32 || reopt_remap_list.size() < 32) {
		reopt_qt = gpuqo_run_idp2_dp<BitmapsetInner, Bitmapset32>(
								gpuqo_algo, new_info, reopt_remap_list);
	} else if (BitmapsetInner::SIZE == 64 || reopt_remap_list.size() < 64) {
		reopt_qt = gpuqo_run_idp2_dp<BitmapsetInner, Bitmapset64>(
								gpuqo_algo, new_info, reopt_remap_list);
	} else {
		reopt_qt = gpuqo_run_idp2_dp<BitmapsetInner, BitmapsetDynamic>(
								gpuqo_algo, new_info, reopt_remap_list);
	}
	// to reopt_qt apo tin proigoumeni sunartisi einai remapped optimized T' 
	// -----LINE 5 REPLACE T' IN T WITH DP(T',G) WHICH IS reopt_qt AS A TEMPORARY TABLE
	QueryTree<BitmapsetInner> *res_qt;
	// EDW EINAI TO CLAUSE GIA NA TELEIWSEI TO WHILE LOOP.
	// an ta iters pou emeinan 25 einai idio me ta relations tou T pou bike stin eisodo tou function
	// tote molis to kaname kai auto optimize ara theloume apla na kanoume return to res_qt
	if (new_info->n_iters == new_info->n_rels){
		res_qt = reopt_qt;
	} else {
		list<remapper_transf_el_t<BitmapsetInner> > next_remap_list;

		// ksanadimiourgei ena akoma remapper gia to pleon optimized subset
		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopt_qt->id;
		// allazei to id tou "head?" tou tree se 0
		list_el.to_idx = 0;
		// kanei attach to optimized query tree
		list_el.qt = reopt_qt;
		next_remap_list.push_back(list_el);
		
		// apo to 1 giati idi exoume valei to head, to thema einai na valoume ta ypoloipa
		int j = 1;
		for (int i=0; i<new_info->n_rels; i++){
			// gia ola ta relations POU DEN EINAI SUBSET tou optimized table rels tou T-T' dhladh, pas kai dimourgeis to neo remapping
			// Basically, the optimized subtree (reopt_qt) takes id=0 in the next_remap_list and all other relations NOT part of the optimized table are
			// then .pushed_back() to the next_remap_list after it.
			if (!new_info->base_rels[i].id.isSubset(reopt_qt->id)){
				list_el.from_relid = new_info->base_rels[i].id;
				list_el.to_idx = j++;
				// only the list_el[0] has a .qt member
				list_el.qt = NULL;
				next_remap_list.push_back(list_el);
			}
		}

		if (BitmapsetInner::SIZE == 32 || next_remap_list.size() < 32) {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, Bitmapset32>(
				gpuqo_algo, new_goo_qt, new_info, next_remap_list, n_iters);
		} else if (BitmapsetInner::SIZE == 64 || next_remap_list.size() < 64) {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, Bitmapset64>(
				gpuqo_algo, new_goo_qt, new_info, next_remap_list, n_iters);
		} else {
			res_qt = gpuqo_run_idp2_rec<BitmapsetInner, BitmapsetDynamic>(
				gpuqo_algo, new_goo_qt, new_info, next_remap_list, n_iters);
		}
	}

	// -----LINE 8: EXPAND ALL TEMPORARY TABLES IN T BACK TO THEIR JOIN TREE
	QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(res_qt);
	// free what we dont need 
	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_goo_qt);
	freeQueryTree(res_qt);
	// and return T
	return out_qt;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_idp2(int gpuqo_algo, 
									GpuqoPlannerInfo<BitmapsetN>* info,
									int n_iters)
{
	// goo_qt einai to full join tree twn relations olou tou query graph
	QueryTree<BitmapsetN> *goo_qt = gpuqo_cpu_goo(info);
	// ftiaxnei mia lista twn remapper_transf_el_t<BitmapsetN>
	// edw vevaia to initialization einai from base_rel.id se i, (mono min exei thema me to -1 pou mou elege ta idx
	// alliws fainetai na einai san remap ston eauto tou)
	list<remapper_transf_el_t<BitmapsetN> > remap_list;
	// initialize remap_list
	for (int i=0; i<info->n_rels; i++){
		// dimiourgei ta elements list_el, tous vazei id index default query tree to NULL
		// to push_bash einai san to append (apo ta de3ia prosthetei stin lista)
		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = info->base_rels[i].id;
		list_el.to_idx = i;
		list_el.qt = NULL;
		remap_list.push_back(list_el);
	}

	//run recursive function, with default idp_n_iters ( I usually give 25)
	QueryTree<BitmapsetN> *out_qt = gpuqo_run_idp2_rec<BitmapsetN,BitmapsetN>(
						gpuqo_algo, goo_qt, info, remap_list, 
						n_iters > 0 ? n_iters : gpuqo_idp_n_iters);
	// recursively deletes all nodes in qt (gpuqo_query_tree.cuh)
	freeQueryTree(goo_qt);

	return out_qt;
}

template QueryTree<Bitmapset32> *gpuqo_run_idp2<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*,int);
template QueryTree<Bitmapset64> *gpuqo_run_idp2<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*,int);
template QueryTree<BitmapsetDynamic> *gpuqo_run_idp2<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*,int);