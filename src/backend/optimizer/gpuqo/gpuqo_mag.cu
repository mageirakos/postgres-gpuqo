/*------------------------------------------------------------------------
 *
 * gpuqo_mag.cu
 *      dp over dp and k-cut implementation
 *
 * src/backend/optimizer/gpuqo/gpuqo_mag.cu
 *
 *-------------------------------------------------------------------------
 */

// 1. Find k-1 edges to cut (helper function?)
//		a) iterate through all the edges of current graph and estimate_cost, estimate_rows
// 2. Create the k disjoint sets by cutting the edges (disjoins since we are only focusing on the tree case)
// 3. Optimize the k sets
//		a) For each optimized query tree, create the compound node
//		b) size(query graph including all compound nodes) > 25 ? recurse : final recurse? 
// 				(look how its handled in idp2)



// !!!! This recursive function is PER LEVEL OF DP, so we recurse after all k-subsets of current level 
// are optimized, turned into compound nodes and creating the query tree

// mag_rec(): 
// does remapping from caller like idp?
// has helper function for (1) (like find_maximal_subtree)
// Loop over all cuts, for left/right nodes dfs or bfs to create get disjoint set mapping 
// 		and send it to optimizer? ( double work if not keeping track of seen set)
// each time we get optimized query tree, keep a list of all the nodes
// at the end we have yet another double for loop and iterate through them to create all compound nodes
// join all the compound nodes into a query graph


#include "gpuqo.cuh" 
#include "gpuqo_query_tree.cuh"
#include "gpuqo_filter.cuh" // grow
#include "gpuqo_row_estimation.cuh" // estimate

#include <iostream>
#include <vector>
#include <string>

#include <algorithm>

// int gpuqo_idp_n_iters;
// int gpuqo_idp_type;

template<typename BitmapsetN>
struct GraphEdge{
	BitmapsetN left;
	BitmapsetN right;
	float rows;
	float selectivity;
};

template<typename BitmapsetN>
struct SortEdges{
    bool operator()(const GraphEdge<BitmapsetN>& lhs, const GraphEdge<BitmapsetN>& rhs) {
        return lhs.rows + lhs.selectivity > rhs.rows + rhs.selectivity;
    }
};

// GLOBAL JUST FOR TESTING
int level_of_dp = 0;

// Improvement 1 : Use Max Heap and pop top k
// Improvement 2 : Return pointer to vector not the vector itself
// Improvement 3 : Improve on the datastructures used (for this I need to see what breaks later)
// Note 1 : I think I need to run analyze command at some before estimating rows/select but idk where
// Note 2 : Working with wrong structs until I figure out how custom estimate and JoinRelations work
template<typename BitmapsetN>
std::vector<GraphEdge<BitmapsetN>> find_k_cut_edges(GpuqoPlannerInfo<BitmapsetN>* info, int k)
{
	printf("\n f(x): find_k_cut_edges \n");
	int *bfs_queue = new int[info->n_rels];
    int bfs_queue_left_idx = 0;
    int bfs_queue_right_idx = 0;

	std::vector<GraphEdge<BitmapsetN>> edge_list;

    int bfs_idx = 0;
    bfs_queue[bfs_queue_right_idx++] = 0;

	// a set with just the 2nd bit set (starts from 0) - which relations have been seen
    BitmapsetN seen = BitmapsetN::nth(1);
	while (bfs_queue_left_idx != bfs_queue_right_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_left_idx++]; 

        BitmapsetN edges = info->edge_table[base_rel_idx];

		GraphEdge<BitmapsetN> edge_el;
		edge_el.left = info->base_rels[base_rel_idx].id;
		float left_rel_rows = info->base_rels[base_rel_idx].rows;

		bfs_idx++;
		while(!edges.empty()){
			int next = edges.lowestPos(); 
			Assert(next > 0);

    		edge_el.right =  info->base_rels[next-1].id; 
			float right_rel_rows = info->base_rels[next-1].rows;

			edge_el.selectivity = estimate_join_selectivity(edge_el.left, edge_el.right, info);
			edge_el.rows = edge_el.selectivity * left_rel_rows * right_rel_rows;
			edge_list.push_back(edge_el);
			
			if(!seen.isSet(next)){
				bfs_queue[bfs_queue_right_idx++] = next - 1;
			}

			edges.unset(next);

		}
		seen |= info->edge_table[base_rel_idx];
	}
	
	
	std::sort(edge_list.begin(), edge_list.end(), SortEdges<BitmapsetN>()); 
	std::vector<GraphEdge<BitmapsetN>> resVec(edge_list.begin(), edge_list.begin() + k);
	delete[] bfs_queue;

	return  resVec;
}


template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_mag_dp(int gpuqo_algo, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	printf("\n f(x): gpuqo_run_mag_dp for LEVEL %d\n", level_of_dp);
	// the segmentation fault is somewhere after here (after we are in the terminal if of recursion)
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);

	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	new_info->n_iters = new_info->n_rels;

	LOG_PROFILE("MAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	printf("MAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_switch(gpuqo_algo, new_info);
	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);
	
	return new_qt_remap;
}


template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_mag_rec(int gpuqo_algo, 
					GpuqoPlannerInfo<BitmapsetOuter>* info,
					list<remapper_transf_el_t<BitmapsetOuter> > &remap_list,
					int n_iters) 
{
	level_of_dp++;
	std::cout << " LEVEL OF DP: " << level_of_dp << std::endl;
	printf("\n f(x): gpuqo_run_mag_rec ----- FIRST CHECK ----- LEVEL %d \n", level_of_dp);
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	
	// std::cout << "BEFORE" << "new_info->n_iters: " << new_info->n_iters << "  n_iters: " << n_iters << "  new_info->n_rels: " << new_info->n_rels << std::endl;	
	new_info->n_iters = min(new_info->n_rels, n_iters);
	// std::cout << "AFTER" << "new_info->n_iters: " << new_info->n_iters << "  n_iters: " << n_iters << "  new_info->n_rels: " << new_info->n_rels << std::endl;	
	

	if (new_info->n_rels == new_info->n_iters){ // its going to be equal because of the above min()
		printf("\n f(x): gpuqo_run_mag_rec => INSIDE termination check at LEVEL OF DP %d \n", level_of_dp);
		// std::cout << "INSIDE TERMINATION CHECK"<< "new_info->n_iters: " << new_info->n_iters << "  n_iters" << n_iters << "  new_info->n_rels" << new_info->n_rels << std::endl;		
		list<remapper_transf_el_t<BitmapsetInner> > remap_list_2;
		// should be remap to itself
		for (int i=0; i<new_info->n_rels; i++){
			remapper_transf_el_t<BitmapsetInner> list_el;
			list_el.from_relid = new_info->base_rels[i].id;
			list_el.to_idx = i;
			list_el.qt = NULL;
			remap_list_2.push_back(list_el);
		}

		printf("\n f(x): gpuqo_run_mag_rec => INSIDE termination check => after BFS \n");
		QueryTree<BitmapsetInner> *reopt_qt;
		// // remap_list from caller is not correct to be given to run_dp
		if (BitmapsetInner::SIZE == 32 || remap_list_2.size() < 32) {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, Bitmapset32>(
									gpuqo_algo, new_info, remap_list_2);
		} else if (BitmapsetInner::SIZE == 64 || remap_list_2.size() < 64) {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, Bitmapset64>(
									gpuqo_algo, new_info, remap_list_2);
		} else {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, BitmapsetDynamic>(
									gpuqo_algo, new_info, remap_list_2);
		}
		printf("\n f(x): gpuqo_run_mag_rec => INSIDE termination check => after opt \n");
		QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(reopt_qt);
		freeGpuqoPlannerInfo(new_info);
		freeQueryTree(reopt_qt);
		printf("\n f(x): gpuqo_run_mag_rec => INSIDE termination check => RETURNING \n");
		return out_qt;
	}

	// #TODO - This needs to scale dynamically not hardcoded
	// #TODO - We need to take into account the size of the subsets created
	int num_edges_to_cut = 2; // k
	std::vector<GraphEdge<BitmapsetInner>> cutEdges = find_k_cut_edges(new_info, num_edges_to_cut);
	printf("FOUND %zu cutEdges", cutEdges.size());
	printf("\n PRINTING EDGES TO CUT\n");
	for(int i=0; i<=num_edges_to_cut; i++){
		std::cout << "edge_" << i << " : left= " << cutEdges[i].left.toUint() << "\tright= " << cutEdges[i].right.toUint()<< std::endl; 
		std::cout << "a.k.a edge: " << cutEdges[i].left.lowestPos()-1 << "-" << cutEdges[i].right.lowestPos()-1 << std::endl;		
	}
		

	printf("\n f(x): gpuqo_run_mag_rec => after find_k_cut_edges ----- CHECK 2 ----- \n");
	// duplicate edge_table
	BitmapsetInner edge_table_copy[BitmapsetInner::SIZE];
	printf("\n edge table BEFORE: \n");
	printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	for (int i=0; i < BitmapsetInner::SIZE; i++){
		std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i].toUint() << std::endl;
		edge_table_copy[i] = new_info->edge_table[i];
	}
	// remove edges from copy
	for (int i=0; i < num_edges_to_cut; i++){  
		GraphEdge<BitmapsetInner> cut_edge = cutEdges[i];
		std::cout << "removing edge: " << cut_edge.left.lowestPos()-1 << "--" << cut_edge.right.lowestPos()-1 << std::endl;
		edge_table_copy[cut_edge.left.lowestPos()-1].unset(cut_edge.right.lowestPos()); 
		edge_table_copy[cut_edge.right.lowestPos()-1].unset(cut_edge.left.lowestPos()); 
	}
	
	printf("\n edge table AFTER: \n");
	printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	for (int i=0; i < BitmapsetInner::SIZE; i++){
		std::cout << "edge_table["<< i <<"] = "<< edge_table_copy[i].toUint() << std::endl;
	}
	
	// based on the prints the removal of the edges is done correctly 
	// on test with k=2 edges and with k=10 edges


	// 3. Add all base_rel_id to a bitmapset (also done in cpusequential) so we grow without duplicates
	BitmapsetInner subset_baserel_id = BitmapsetInner(0);
	for (int i = 0; i < new_info->n_rels; i++){
		subset_baserel_id |= new_info->base_rels[i].id;
	}

	// get subgraphs
	// printf(" PRINTING SUBGRAPHS: \n");
	std::vector<BitmapsetInner> subgraphs;
	while(subset_baserel_id.toUint()!=0){
		BitmapsetInner csg = grow(subset_baserel_id.lowest(), subset_baserel_id, edge_table_copy);
		// std::cout << "before: " << subset_baserel_id.toUint() << " from csg: " << csg.toUint() << std::endl; 		
		subset_baserel_id = subset_baserel_id.differenceSet(csg);
		// std::cout << "after: " << subset_baserel_id.toUint() << " from csg: " << csg.toUint() << std::endl; 
		subgraphs.push_back(csg);
		std::cout << "csg_" << subgraphs.size() << " : " << csg.toUint() << " at LEVEL " << level_of_dp <<std::endl; 
	}


// UP TO HERE SUBGRAPHS ARE CORRECTLY CREATED, EDGES ARE CORRECTLY CUT ( I STILL NEED TO CHECK THE RANKING OF (RS) WHEN CURRING EDGES AND EVERYTHING BELOW THIS LINE)

	// we are now left with a vector of the subgraphs which we want to optimize

	printf("\n f(x): gpuqo_run_mag_rec => after subgraphs ----- CHECK 3 ----- \n");
	// dp over subgraphs	
	list<remapper_transf_el_t<BitmapsetInner> > next_remap_list;
	for (int i=0; i < subgraphs.size(); i++){
		// bfs naming for each subgrap
		list<remapper_transf_el_t<BitmapsetInner> > reopt_remap_list;
		int j = 0;
		BitmapsetInner reopTables = subgraphs[i];
		while (!reopTables.empty()) {
			remapper_transf_el_t<BitmapsetInner> list_el;
			list_el.from_relid = reopTables.lowest();
			list_el.to_idx = j++;
			list_el.qt = NULL;
			reopt_remap_list.push_back(list_el);
			reopTables -= list_el.from_relid;
		}
		printf("\n f(x): gpuqo_run_mag_rec => opt for loop ----- CHECK 3.1 ----- \n");
		
		// optimize
		QueryTree<BitmapsetInner> *reopt_qt;
		if (BitmapsetInner::SIZE == 32 || reopt_remap_list.size() < 32) {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, Bitmapset32>(
									gpuqo_algo, new_info, reopt_remap_list);
		} else if (BitmapsetInner::SIZE == 64 || reopt_remap_list.size() < 64) {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, Bitmapset64>(
									gpuqo_algo, new_info, reopt_remap_list);
		} else {
			reopt_qt = gpuqo_run_mag_dp<BitmapsetInner, BitmapsetDynamic>(
									gpuqo_algo, new_info, reopt_remap_list);
		}
		printf("\n f(x): gpuqo_run_mag_rec => opt for loop ----- CHECK 3.2 ----- \n");
		
		// composite node
		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopt_qt->id;
		list_el.to_idx = i;
		list_el.qt = reopt_qt;
		next_remap_list.push_back(list_el);
		printf("\n f(x): gpuqo_run_mag_rec => opt for loop ----- CHECK 3.3 ----- \n");
	}


	printf("\n f(x): gpuqo_run_mag_rec => calling recursion ----- CHECK 4 ----- LEVEL %d \n", level_of_dp);
	// recursion
	QueryTree<BitmapsetInner> *res_qt;
	if (BitmapsetInner::SIZE == 32 || next_remap_list.size() < 32) {
		res_qt = gpuqo_run_mag_rec<BitmapsetInner, Bitmapset32>(
		gpuqo_algo, new_info, next_remap_list, n_iters);
	} else if (BitmapsetInner::SIZE == 64 || next_remap_list.size() < 64) {
		res_qt = gpuqo_run_mag_rec<BitmapsetInner, Bitmapset64>(
		gpuqo_algo, new_info, next_remap_list, n_iters);
	} else {
		res_qt = gpuqo_run_mag_rec<BitmapsetInner, BitmapsetDynamic>(
			gpuqo_algo, new_info, next_remap_list, n_iters);
	}
	printf("\n f(x): gpuqo_run_mag_rec => after recursion ----- CHECK 5 ----- \n");
	QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(res_qt);
	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(res_qt);
	printf("\n f(x): gpuqo_run_mag_rec => returning ----- FINAL CHECK ----- LEVEL %d \n", level_of_dp);
	return out_qt;
}


template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_mag(int gpuqo_algo, 
									GpuqoPlannerInfo<BitmapsetN>* info,
									int n_iters)
{
	printf("\n f(x): gpu_run_mag \n");
	list<remapper_transf_el_t<BitmapsetN> > remap_list;

	for (int i=0; i<info->n_rels; i++){
		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = info->base_rels[i].id;
		list_el.to_idx = i;
		list_el.qt = NULL;
		remap_list.push_back(list_el);
	}

	// recursive function
	QueryTree<BitmapsetN> *out_qt = gpuqo_run_mag_rec<BitmapsetN,BitmapsetN>(
						gpuqo_algo, info, remap_list, 
						n_iters > 0 ? n_iters : gpuqo_idp_n_iters);

	return out_qt;
}

template QueryTree<Bitmapset32> *gpuqo_run_mag<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*,int);
template QueryTree<Bitmapset64> *gpuqo_run_mag<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*,int);
template QueryTree<BitmapsetDynamic> *gpuqo_run_mag<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*,int);

