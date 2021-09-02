/*------------------------------------------------------------------------
 *
 * gpuqo_mag.cu
 *      dp over dp and k-cut implementation
 *
 * src/backend/optimizer/gpuqo/gpuqo_mag.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo.cuh" 
#include "gpuqo_query_tree.cuh"
#include "gpuqo_filter.cuh" // grow
#include "gpuqo_row_estimation.cuh" // estimate
#include "gpuqo_debug.cuh" // print bms
#include "gpuqo_bitmapset.cuh"
#include "gpuqo_bitmapset_dynamic.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

static int level_of_dp = 0;
int gpuqo_k_cut_edges;

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

// template<typename BitmapsetN>
// int getNumberOfEdges(GpuqoPlannerInfo<BitmapsetN>* info){
//     int number_of_edges = 0 ;
//     for(int i=0; i < info->n_rels; i++){
//         number_of_edges += info->edge_table[i].size();
//     }
//     assert(number_of_edges%2==0); // should be even because undirected graph
//     return number_of_edges/2;
// };


// #include "kaHIP_interface.h"
// static int test() {

        // std::cout <<  "partitioning graph from the manual"  << std::endl;

        // int n            = 5;
        // int* xadj        = new int[6];
        // xadj[0] = 0; xadj[1] = 2; xadj[2] = 5; xadj[3] = 7; xadj[4] = 9; xadj[5] = 12;

        // int* adjncy      = new int[12];
        // adjncy[0]  = 1; adjncy[1]  = 4; adjncy[2]  = 0; adjncy[3]  = 2; adjncy[4]  = 4; adjncy[5]  = 1; 
        // adjncy[6]  = 3; adjncy[7]  = 2; adjncy[8]  = 4; adjncy[9]  = 0; adjncy[10] = 1; adjncy[11] = 3; 
        
        // double imbalance = 0.03;
        // int* part        = new int[n];
        // int edge_cut     = 0;
        // int nparts       = 3;
        // int* vwgt        = NULL;
        // int* adjcwgt     = NULL;

        // kaffpa(&n, vwgt, xadj, adjcwgt, adjncy, &nparts, &imbalance, false, 0, ECO, & edge_cut, part);

        // std::cout <<  "edge cut " <<  edge_cut  << std::endl;
		// for(int i=0; i < n; i++){
		// 	std::cout << "blockID[" << i << "] = " << part[i] << std::endl;
		// }
		// printf("\nentering infinite loop;\n");
		// while(1==1){
			
		// }
// }


template<typename BitmapsetN>
std::vector<GraphEdge<BitmapsetN>> find_k_cut_edges(GpuqoPlannerInfo<BitmapsetN>* info, int k)
{
	printf("\n f(x): find_k_cut_edges \n");
	int *bfs_queue = new int[info->n_rels];
    int bfs_queue_left_idx = 0;
    int bfs_queue_right_idx = 0;

	// int number_of_edges = getNumberOfEdges(info); // m (only count once, don't count duplicate edges)
	// printf("%d", number_of_edges);
	// std::cout << "number of edges: " << number_of_edges << std::endl;
	// while(1){}
	// test();
	std::vector<GraphEdge<BitmapsetN>> edge_list;

    int bfs_idx = 0;
    bfs_queue[bfs_queue_right_idx++] = 0;

	// a set with just the 2nd bit set (starts from 0) - which relations have been seen
    BitmapsetN seen = BitmapsetN::nth(1);
	while (bfs_queue_left_idx != bfs_queue_right_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_left_idx++]; 

        BitmapsetN edges = info->edge_table[base_rel_idx];

		bfs_idx++;
		while(!edges.empty()){
			int next = edges.lowestPos(); 
			Assert(next > 0);
			if(!seen.isSet(next)){
				bfs_queue[bfs_queue_right_idx++] = next - 1;

				GraphEdge<BitmapsetN> edge_el;
				edge_el.left = info->base_rels[base_rel_idx].id;
				float left_rel_rows = info->base_rels[base_rel_idx].rows;
				edge_el.right =  info->base_rels[next-1].id; 
				float right_rel_rows = info->base_rels[next-1].rows;
				edge_el.selectivity = estimate_join_selectivity(edge_el.left, edge_el.right, info);
				edge_el.rows = edge_el.selectivity * left_rel_rows * right_rel_rows;

				edge_list.push_back(edge_el);
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

	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	new_info->n_iters = new_info->n_rels;

	LOG_PROFILE("MAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	printf("\nMAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
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
	std::cout << "\n\t LEVEL OF DP: " << level_of_dp << std::endl;
	printf("\n f(x): gpuqo_run_mag_rec ----- FIRST CHECK ----- LEVEL %d \n", level_of_dp);
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	
	new_info->n_iters = min(new_info->n_rels, n_iters);

	if (new_info->n_rels == new_info->n_iters){ // its going to be equal because of the above min()
		printf("\n f(x): gpuqo_run_mag_rec => -----INSIDE TERMINATION CHECK ----- LEVEL %d \n", level_of_dp);
		std::cout << "new_info->n_iters: " << new_info->n_iters << "  n_iters: " << n_iters << "  new_info->n_rels: " << new_info->n_rels << std::endl;		
		list<remapper_transf_el_t<BitmapsetInner> > remap_list_2;
		
		// should be remap to itself
		for (int i=0; i<new_info->n_rels; i++){
			remapper_transf_el_t<BitmapsetInner> list_el;
			list_el.from_relid = new_info->base_rels[i].id;
			list_el.to_idx = i;
			list_el.qt = NULL;
			remap_list_2.push_back(list_el);
		}

		QueryTree<BitmapsetInner> *reopt_qt;
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
		QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(reopt_qt);
		freeGpuqoPlannerInfo(new_info);
		freeQueryTree(reopt_qt);
		printf("\n f(x): gpuqo_run_mag_rec => returning -----INSIDE TERMINATION CHECK -----\n");
		return out_qt;
	}


	// FIX: This probably needs to change as we expand to normal graphs (other than trees)
	// in trees, edges are at most 1 less than number of nodes, so when I use a large k we should be careful 
	// This FIX does not seem to work ????
	std::cout << "BEFORE gpuqo_k_cut_edges = " << gpuqo_k_cut_edges << " new_info->n_rels= " << new_info->n_rels << std::endl;
	if (gpuqo_k_cut_edges >= new_info->n_rels-1){ // -1 because infinite loop since we won't decrease at all
		gpuqo_k_cut_edges = new_info->n_rels - 20; // -20 so that graph decreases by at least 10 node each time
	} 
	std::cout << "AFTER gpuqo_k_cut_edges = " << gpuqo_k_cut_edges << " new_info->n_rels= " << new_info->n_rels << std::endl;



	std::vector<GraphEdge<BitmapsetInner>> cutEdges = find_k_cut_edges(new_info, gpuqo_k_cut_edges);
	printf("FOUND %zu cutEdges", cutEdges.size());
	printf("\tPRINTING  CUT EDGES\n, where 1st rel(2^1) = 1 (start counting from 1 in rel id)");
	for(int i=0; i<gpuqo_k_cut_edges; i++){
		std::cout << "edge_" << i << " : left= " << cutEdges[i].left.toUlonglong() << "\tright= " << cutEdges[i].right.toUlonglong() \
		<<  "\t(" << cutEdges[i].left.lowestPos() << "-" << cutEdges[i].right.lowestPos()-1 << ") " << std::endl;		
	}

	printf("\n f(x): gpuqo_run_mag_rec => after find_k_cut_edges ----- CHECK 2 ----- \n");
	BitmapsetInner edge_table_copy[BitmapsetInner::SIZE];
	printf("\n edge table BEFORE: \n");
	printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	std::cout << "                 3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	for (int i=0; i < new_info->n_rels; i++){
		// std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i].toUlonglong() << std::endl;
		std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i] << std::endl;
		edge_table_copy[i] = new_info->edge_table[i];
	}

	for (int i=0; i < gpuqo_k_cut_edges; i++){  
		GraphEdge<BitmapsetInner> cut_edge = cutEdges[i];
		std::cout << "removing edge: " << cut_edge.left.lowestPos() << "--" << cut_edge.right.lowestPos() << " - 1st rel(2^1) = 1" << std::endl;
		edge_table_copy[cut_edge.left.lowestPos()-1].unset(cut_edge.right.lowestPos()); 
		edge_table_copy[cut_edge.right.lowestPos()-1].unset(cut_edge.left.lowestPos()); 
	}
	
	// Lopp only for printing edge_table AFTER
	printf("\n edge table AFTER: \n");
	printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	for (int i=0; i < new_info->n_rels; i++){
		std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i] << std::endl;
	}
	
	printf("\nsubset_baserel_id BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	BitmapsetInner subset_baserel_id = BitmapsetInner(0);
	for (int i = 0; i < new_info->n_rels; i++){
		std::cout << "iter: " << i << " new_info->base_rels[" << i << "].id = " << new_info->base_rels[i].id.toUlonglong() << std::endl;
		subset_baserel_id |= new_info->base_rels[i].id;
	}
	std::cout << "                  3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	std::cout << "subset_baserel_id " << subset_baserel_id << std::endl;

	
	printf("\nPRINTING SUBGRAPHS: \n");
	std::vector<BitmapsetInner> subgraphs;
	std::cout << " 3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	while(subset_baserel_id!=0){
		BitmapsetInner csg = grow(subset_baserel_id.lowest(), subset_baserel_id, edge_table_copy);
		// std::cout << "before: " << subset_baserel_id << " from csg: " << csg << std::endl; 		
		std::cout << "before : " << subset_baserel_id.toUlonglong() << " from csg: " << csg.toUlonglong() << std::endl; 
		subset_baserel_id = subset_baserel_id.differenceSet(csg);
		std::cout << "after : " << subset_baserel_id.toUlonglong() << " from csg: " << csg.toUlonglong() << std::endl; 
		// std::cout << "after bms: " << subset_baserel_id << " from csg: " << csg<< std::endl; 		
		subgraphs.push_back(csg);
		std::cout << "csg_" << subgraphs.size() << " : " << csg.toUlonglong() <<std::endl; 
		std::cout << subset_baserel_id << std::endl;
	}

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
		
		// composite node
		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopt_qt->id;
		list_el.to_idx = i;
		list_el.qt = reopt_qt;
		next_remap_list.push_back(list_el);
	}


	printf("\n f(x): gpuqo_run_mag_rec => recursing ----- CHECK 4 ----- LEVEL %d \n", level_of_dp);
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
	printf("\n f(x): gpuqo_run_mag_rec => after recursion ----- CHECK 5 ----- LEVEL %d \n", level_of_dp);
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

	// reset it for additional runs 
	level_of_dp = 0;
	return out_qt;
}

template QueryTree<Bitmapset32> *gpuqo_run_mag<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*,int);
template QueryTree<Bitmapset64> *gpuqo_run_mag<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*,int);
template QueryTree<BitmapsetDynamic> *gpuqo_run_mag<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*,int);

