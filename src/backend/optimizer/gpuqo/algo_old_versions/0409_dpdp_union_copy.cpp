/*------------------------------------------------------------------------
 *
 * gpuqo_dpdp_union.cu
 *      dp over dp and k-cut implementation
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpdp_union.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo.cuh" 
#include "gpuqo_query_tree.cuh"
// #include "gpuqo_filter.cuh" // grow
#include "gpuqo_row_estimation.cuh" // estimate
#include "gpuqo_debug.cuh" // print bms
#include "gpuqo_bitmapset.cuh"
#include "gpuqo_bitmapset_dynamic.cuh"

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <algorithm> // std::max
#include <queue> // priority queue
#include <unordered_map> // used for disjoint sets

static int level_of_dp = 0;

template<typename BitmapsetN>
class DisjointSet
{
    std::unordered_map<BitmapsetN, BitmapsetN> parent;
    std::unordered_map<BitmapsetN, int> size;
	//TODO: Test if csgs are correct
    std::unordered_map<BitmapsetN, BitmapsetN> csg;

public:
	//TODO: Test if csgs are correct
    BitmapsetN getCsg(BitmapsetN node_id){
        return csg[Find(node_id)];
    }
    
	int getSize(BitmapsetN node_id){
        return size[Find(node_id)];
    }

    void makeSet(GpuqoPlannerInfo<BitmapsetN>* info) 
    {
        // create `n` disjoint sets (one for each node)
        for (int base_rel_idx=0; base_rel_idx<info->n_rels; base_rel_idx++){
            BitmapsetN node_id = info->base_rels[base_rel_idx].id;
            parent[node_id] = node_id;
            csg[node_id] = node_id; // csg starts out with just itself
            size[node_id] = 1;
        }
    }
 
    BitmapsetN Find(BitmapsetN node_id)
    {
        if (parent[node_id] != node_id)
        {
            parent[node_id] = Find(parent[node_id]);
        }
 
        return parent[node_id];
    }
 
    void Union(BitmapsetN node_a, BitmapsetN node_b)
    {
        BitmapsetN x = Find(node_a);
        BitmapsetN y = Find(node_b);
        
        if (x == y) { 
            return;
        }

        if (size[x] > size[y]) {
            parent[y] = x;
            size[x] += size[y];
			//TODO: Test csg
            csg[x] |= csg[y];          
        }
        else if (size[x] < size[y]) {
            parent[x] = y;
            size[y] += size[x];
			//TODO: Test csg
            csg[y] |= csg[x];
        }
        else {
            parent[x] = y;
			// sizes were equal so it doubles
            size[y] *= 2; 
			//TODO: Test csg
            csg[y] |= csg[x];
        }
    }
};

template<typename BitmapsetN>
void printSets(GpuqoPlannerInfo<BitmapsetN>* info, DisjointSet<BitmapsetN> &ds)
{
    for (int i=0;i<info->n_rels;i++) {
		BitmapsetN node_id = info->base_rels[i].id;
		printf("\n");
        std::cout << ds.Find(node_id) << ' ';
        std::cout << 'size: ( ' << ds.getSize(node_id) << ' )' << '-';
        std::cout << 'csg: ( ' << ds.getCsg(node_id) << ' )' << '-';
    }
    std::cout << std::endl;
};

/// ------------------------------- END OF DISJOINT SET CLASS

//NOTE: This GraphEdge is different from the _dpdp and _mag implementations 
// It is customized for the _dpdp_union algorithm
template<typename BitmapsetN>
struct GraphEdge {
	BitmapsetN left;
	BitmapsetN right;
	//TODO: create these on BFS
	int left_size; // union size
	int right_size; // union size
	int total_size; // left + right
	float rows;
	float selectivity;
	//TODO: create this ON BFS
	float weight; // whatever combination I choose
};

template<typename BitmapsetN>
struct CompareLeafEdges{
	// > is mean heap because default is max heap (<)
    bool operator()(const GraphEdge<BitmapsetN>* lhs, const GraphEdge<BitmapsetN>* rhs) {
		return lhs->weight > rhs->weight;
    }
};

template<typename BitmapsetN>
struct CompareEdges{
	// > is mean heap because default is max heap (<)
    bool operator()(const GraphEdge<BitmapsetN>* lhs, const GraphEdge<BitmapsetN>* rhs) {
		return lhs->total_size > lhs->total_size || (lhs->total_size == lhs->total_size  && lhs->weight > rhs->weight);
    }
};

//TODO: Should these be BitmapsetInner? or BitmapN
template<typename BitmapsetN>
using LeafQ = std::priority_queue<GraphEdge<BitmapsetN>*, std::vector<GraphEdge<BitmapsetN>*>, CompareLeafEdges<BitmapsetN>>;
template<typename BitmapsetN>
//TODO : Change back to CompareEdges
using EdgeQ = std::priority_queue<GraphEdge<BitmapsetN>*, std::vector<GraphEdge<BitmapsetN>*>, CompareEdges<BitmapsetN>>;


//TODO: Test if correct weight
template<typename BitmapsetN>
GraphEdge<BitmapsetN>* createGraphEdge(int left_rel_idx, int right_rel_idx , GpuqoPlannerInfo<BitmapsetN>* info){

	GraphEdge<BitmapsetN>* edge_el = new GraphEdge<BitmapsetN>;

	edge_el->left = info->base_rels[left_rel_idx].id;
	edge_el->right =  info->base_rels[right_rel_idx].id; 

	float left_rel_rows = info->base_rels[left_rel_idx].rows;
	float right_rel_rows = info->base_rels[right_rel_idx].rows;
	
	edge_el->selectivity = estimate_join_selectivity(edge_el->left, edge_el->right, info);
	edge_el->rows = edge_el->selectivity * left_rel_rows * right_rel_rows;

	// std::cout << "join selectivity edge " << left_rel_idx << "-" << right_rel_idx <<  " = " << edge_el->selectivity << std::endl;
	// std::cout << "join cardinality(rows) edge " << left_rel_idx << "-" << right_rel_idx <<  " = " << edge_el->rows << std::endl;

	//TODO: Improve weight (+ add cost)
	// selectivity is a number from 0 to 1 thus from below (rows dominates)
	// normalize(rows): row_i' = [row_i - min(rows)] / [max(rows) - min(rows)]

	// edge_el->weight = edge_el->selectivity + edge_el->rows;
	edge_el->weight = edge_el->rows;

	// Disjoint sets not created when BFS/this function is run so just add left/right size = 1. 
	edge_el->left_size = 1;
	edge_el->right_size = 1;
	edge_el->total_size = edge_el->left_size + edge_el->right_size;

	return edge_el;
}


//TODO: Change BFS to incorporate what I need for the UNION algorithm
// (a) Assign edge weights
// (b) Find leaves and initialize the LeafPriorityQueue
// (c) Initialize EdegePriorityQueue (with all edges even leaf ones) - updates will be handled later
//TODO: Make sure this function returns 2 priority queues
//TODO: FIX PRIORITY QUEUE DECLARATIONS ON FUNCTION AND HOW TO DO WHAT I WANT
template<typename BitmapsetN>
void fillPriorityQueues(std::vector<GraphEdge<BitmapsetN>*> &edge_pointers_list, LeafQ<BitmapsetN> &LeafPriorityQueue, EdgeQ<BitmapsetN> &EdgePriorityQueue, GpuqoPlannerInfo<BitmapsetN>* info)
{
	// printf("\n f(x): fillPriorityQueues \n");
	int *bfs_queue = new int[info->n_rels];
    int bfs_queue_left_idx = 0;
    int bfs_queue_right_idx = 0;
	
    int bfs_idx = 0;
    bfs_queue[bfs_queue_right_idx++] = 0;

    BitmapsetN seen = BitmapsetN::nth(1);
	while (bfs_queue_left_idx != bfs_queue_right_idx && bfs_idx < info->n_rels){
        int base_rel_idx = bfs_queue[bfs_queue_left_idx++]; 
		// edges holds the edges to other nodes of the node we are currently traversing
        BitmapsetN edges = info->edge_table[base_rel_idx];

		bfs_idx++;
		while(!edges.empty()){
			int next = edges.lowestPos(); 
			Assert(next > 0);
			//TODO: Does isSet(unsinged) cast into something that I dont want here does logic break?
			if(!seen.isSet(next)){
				bfs_queue[bfs_queue_right_idx++] = next - 1;
				// (a) 
				GraphEdge<BitmapsetN> *edge_el = createGraphEdge(base_rel_idx, next-1 , info);
				// (b)
				// edges is for the edge.left (so the leaf node is always edge.left)
				//TODO TEST (f): Not using LeafPQ
				if (edges.size() == 1) 
				{
					LeafPriorityQueue.push(edge_el);
					// std::cout << "(pushing) LeafQ size = " << LeafPriorityQueue.size() << std::endl;
				}
				// (c) 
				EdgePriorityQueue.push(edge_el);
				// std::cout << "(pushing) EdgeQ size = " << EdgePriorityQueue.size() << std::endl;
				edge_pointers_list.push_back(edge_el);
			}
			edges.unset(next);
		}
		seen |= info->edge_table[base_rel_idx];
	}

	delete[] bfs_queue;
}

template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_dpdp_union_dp(int gpuqo_algo, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	printf("\n f(x): gpuqo_run_dpdp_union_dp for LEVEL %d\n", level_of_dp);

	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	new_info->n_iters = new_info->n_rels;

	LOG_PROFILE("MAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	// printf("\nMAG iteration (dp) with %d rels (%d bits)\n", new_info->n_rels, BitmapsetInner::SIZE);
	QueryTree<BitmapsetInner> *new_qt = gpuqo_run_switch(gpuqo_algo, new_info);
	QueryTree<BitmapsetOuter> *new_qt_remap = remapper.remapQueryTree(new_qt);

	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(new_qt);
	
	return new_qt_remap;
}


template<typename BitmapsetOuter, typename BitmapsetInner>
QueryTree<BitmapsetOuter> *gpuqo_run_dpdp_union_rec(int gpuqo_algo, 
					GpuqoPlannerInfo<BitmapsetOuter>* info,
					list<remapper_transf_el_t<BitmapsetOuter> > &remap_list,
					int n_iters) 
{
	level_of_dp++;
	// std::cout << "\n\t LEVEL OF DP: " << level_of_dp << std::endl;
	// printf("\n f(x): gpuqo_run_dpdp_union_rec ----- FIRST CHECK ----- LEVEL %d \n", level_of_dp);
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	
	// an ta relation einai 26?
	new_info->n_iters = min(new_info->n_rels, n_iters);

	if (new_info->n_rels == new_info->n_iters){ // its going to be equal because of the above min()
		// printf("\n f(x): gpuqo_run_dpdp_union_rec => -----INSIDE TERMINATION CHECK ----- LEVEL %d \n", level_of_dp);
		// std::cout << "new_info->n_iters: " << new_info->n_iters << "  n_iters: " << n_iters << "  new_info->n_rels: " << new_info->n_rels << std::endl;		
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
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, Bitmapset32>(
									gpuqo_algo, new_info, remap_list_2);
		} else if (BitmapsetInner::SIZE == 64 || remap_list_2.size() < 64) {
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, Bitmapset64>(
									gpuqo_algo, new_info, remap_list_2);
		} else {
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, BitmapsetDynamic>(
									gpuqo_algo, new_info, remap_list_2);
		}
		QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(reopt_qt);
		freeGpuqoPlannerInfo(new_info);
		freeQueryTree(reopt_qt);
		// printf("\n f(x): gpuqo_run_dpdp_union_rec => returning -----INSIDE TERMINATION CHECK -----\n");
		return out_qt;
	}


	//TODO: 1) to call the BFS function
	// (a) which initializes LeafPriorityQueue and EdgePriorityQueue
	//TODO: FIX THIS AND HOW TO CLAL THE BFS()

	LeafQ<BitmapsetInner> LeafPriorityQueue;
	EdgeQ<BitmapsetInner> EdgePriorityQueue;
	
	// std::cout << "LeafQ size = " << LeafPriorityQueue.size() << std::endl;
	// printf("After initialization of LeafPriorityQueue and EdgePriorityQueue\n");
	// std::priority_queue<GraphEdge<BitmapsetInner>, std::vector<GraphEdge<BitmapsetInner>>, decltype(CompareLeafEdges<BitmapsetInner>)> LeafPriorityQueue;
	// std::priority_queue<GraphEdge<BitmapsetInner>, std::vector<GraphEdge<BitmapsetInner>>, decltype(CompareEdges<BitmapsetInner>)> EdgePriorityQueue;
	//TODO: Theloun ta <BitmapsetInner> etsi? afou to GraphEdge pou kratane exei <BitmapsetInner>?
	std::vector<GraphEdge<BitmapsetInner>*> edge_pointers_list;
	fillPriorityQueues(edge_pointers_list, LeafPriorityQueue, EdgePriorityQueue, new_info);
	// std::cout << "LeafQ size = " << LeafPriorityQueue.size() << std::endl;
	// printf("After f(x) fillPriorityQueues\n");
	//TODO: 2) Create DisjointSet out of all the nodes in our graph
	
	DisjointSet<BitmapsetInner> ds;
	ds.makeSet(new_info);
	int total_disjoint_sets = new_info->n_rels;
	// printf("After f(x) ds.makeSet()\n");
	// -----------
	//TODO: Improve upper_threshold
	//TODO: Should we add lower_threshold or does it exist because of the algorithm
	// int eps = 0.1; // 10% variation allowed
	// int upper_threshold = 25 + 25*eps;
	int upper_threshold = 25;

	// 3) Create while loop over leaf priority queue for the first set of UNIONs
	// get lowest weight edge of leaf node (edge.left)
	// union with edge.right if it fits on disjoint set
	// printf("Starting LeafPriorityQueue while loop\n");

	//TODO TEST (f): Commenting out the leafPQ while loop
	// int num_leaves = LeafPriorityQueue.size();
	// int worst_leaf_edges = (int)(num_leaves*0.01 + 0.5);
	// while(LeafPriorityQueue.size() > worst_leaf_edges){
	while(!LeafPriorityQueue.empty()){
		const GraphEdge<BitmapsetInner>* edge = LeafPriorityQueue.top();
		LeafPriorityQueue.pop();
		if (ds.getSize(edge->right) + 1 <= upper_threshold){
			ds.Union(edge->left, edge->right);
			total_disjoint_sets--;
		}
	}

	// 4) Create while loop over edge priority queue for the rest of the UNIONs

	// printf("Starting EdgePriorityQueue while loop\n");
	//TODO TEST (f-e): This is what it was
	// I want the ratio to be larger when n_rls "small" and smaller when n_rls "large"
	// at 1000 joins we cut nothing and join everything
	// and as the query graph gets smaller we cut at most up to 50% of the edges

	// creates infinite loop
	// float norm_size = std::max( 0.5, (double)(EdgePriorityQueue.size() - 30) / 970);
	// float cut_ratio = 1 - norm_size; 
	// int worst_inner_edges = std::floor((EdgePriorityQueue.size()) * cut_ratio);
	// std::cout << "norm_size = " << norm_size << " cut_ratio = " << cut_ratio << " worst_inner_edges = " << worst_inner_edges \
	// << " EdgePriorityQueue.size() = " << EdgePriorityQueue.size() << std::endl;
	// while(EdgePriorityQueue.size() >= worst_inner_edges){
		
		// std::cout << "norm_size = " << norm_size << " cut_ratio = " << cut_ratio << " worst_inner_edges = " << worst_inner_edges \
		// << " EdgePriorityQueue.size() = " << EdgePriorityQueue.size() << std::endl;
	// while(EdgePriorityQueue.size() > 20){
	while(!EdgePriorityQueue.empty()){
		GraphEdge<BitmapsetInner>* edge = EdgePriorityQueue.top();
		EdgePriorityQueue.pop();
		// if on different disjoint sets
		if (ds.Find(edge->left) != ds.Find(edge->right) ){
		// 	//TODO TEST: Alternative version of not recalculating size:
		// 	if(ds.getSize(edge->left) + ds.getSize(edge->right) < upper_threshold){
		// 		ds.Union(edge->left, edge->right);
		// 		total_disjoint_sets--;
		// 	}
			// if total size of edge is outdated update and push back into queue
			// happens if either one of the two nodes now belongs in a different disjoint set than when we pushed the edge
			if (edge->total_size != (ds.getSize(edge->left) + ds.getSize(edge->right)) ){
				edge->left_size = ds.getSize(edge->left);
				edge->right_size = ds.getSize(edge->right);
				edge->total_size = edge->left_size + edge->right_size;
				// TODO: kapoio update sto weight an valoume improvement se weights kathws au3anetai to union
				EdgePriorityQueue.push(edge); 
			}
			else{
				if (edge->total_size <= upper_threshold)
				{
					ds.Union(edge->left, edge->right);
					total_disjoint_sets--;
				}
			}
		}
	}
	// //TODO: Do I need the GraphEdges anymore???
	for (int i=0; i < edge_pointers_list.size(); i++){
		delete edge_pointers_list[i];
	}
	// printf("\nAfter EdgeQ while loop\n");
	// TODO: Remove this - just for testing
	// printf("\n\t\tPRINTING ALL DISJOINT SETS");
	// printSets(new_info, ds);

	//5) Get csgs from all the UNIONs (this will be the std::vector<BitmapsetInner> subgraphs;)
	//TODO: Find better way to get csg instead of going through all the nodes again and checking if we've seen them
	// 			- maybe from total_disjoint_sets idk
	std::vector<BitmapsetInner> subgraphs;
	BitmapsetInner seen = BitmapsetInner(0);
	// printf("\n f(x): gpuqo_run_dpdp_union_rec => getting csgs ----- CHECK 2 ----- \n");
	for(int i=0; i<new_info->n_rels; i++)
	{
		BitmapsetInner node_id = new_info->base_rels[i].id;
		BitmapsetInner csg = ds.getCsg(node_id);
		if (!csg.isSubset(seen)){
			subgraphs.push_back(csg);
		}
		seen |= csg;
	}
	// std::cout << "Assertion: subgraphs.size() = " <<  subgraphs.size() << "  total_disjoint_sets= " << total_disjoint_sets << std::endl;
	Assert(subgraphs.size() == total_disjoint_sets);
	
	// 6) optimize/composite nodes recurse should be the same
	// printf("\n f(x): gpuqo_run_dpdp_union_rec => after subgraphs ----- CHECK 3 ----- \n");
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
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, Bitmapset32>(
									gpuqo_algo, new_info, reopt_remap_list);
		} else if (BitmapsetInner::SIZE == 64 || reopt_remap_list.size() < 64) {
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, Bitmapset64>(
									gpuqo_algo, new_info, reopt_remap_list);
		} else {
			reopt_qt = gpuqo_run_dpdp_union_dp<BitmapsetInner, BitmapsetDynamic>(
									gpuqo_algo, new_info, reopt_remap_list);
		}
		
		// composite node
		remapper_transf_el_t<BitmapsetInner> list_el;
		list_el.from_relid = reopt_qt->id;
		list_el.to_idx = i;
		list_el.qt = reopt_qt;
		next_remap_list.push_back(list_el);
	}


	// printf("\n f(x): gpuqo_run_dpdp_union_rec => recursing ----- CHECK 4 ----- LEVEL %d \n", level_of_dp);
	// recursion
	QueryTree<BitmapsetInner> *res_qt;
	if (BitmapsetInner::SIZE == 32 || next_remap_list.size() < 32) {
		res_qt = gpuqo_run_dpdp_union_rec<BitmapsetInner, Bitmapset32>(
		gpuqo_algo, new_info, next_remap_list, n_iters);
	} else if (BitmapsetInner::SIZE == 64 || next_remap_list.size() < 64) {
		res_qt = gpuqo_run_dpdp_union_rec<BitmapsetInner, Bitmapset64>(
		gpuqo_algo, new_info, next_remap_list, n_iters);
	} else {
		res_qt = gpuqo_run_dpdp_union_rec<BitmapsetInner, BitmapsetDynamic>(
			gpuqo_algo, new_info, next_remap_list, n_iters);
	}
	// printf("\n f(x): gpuqo_run_dpdp_union_rec => after recursion ----- CHECK 5 ----- LEVEL %d \n", level_of_dp);
	QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(res_qt);
	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(res_qt);
	// printf("\n f(x): gpuqo_run_dpdp_union_rec => returning ----- FINAL CHECK ----- LEVEL %d \n", level_of_dp);
	return out_qt;
}


template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_dpdp_union(int gpuqo_algo, 
									GpuqoPlannerInfo<BitmapsetN>* info,
									int n_iters)
{
	// printf("\n f(x): gpu_run_dpdp_union \n");
	list<remapper_transf_el_t<BitmapsetN> > remap_list;

	for (int i=0; i<info->n_rels; i++){
		remapper_transf_el_t<BitmapsetN> list_el;
		list_el.from_relid = info->base_rels[i].id;
		list_el.to_idx = i;
		list_el.qt = NULL;
		remap_list.push_back(list_el);
	}

	// recursive function
	QueryTree<BitmapsetN> *out_qt = gpuqo_run_dpdp_union_rec<BitmapsetN,BitmapsetN>(
						gpuqo_algo, info, remap_list, 
						n_iters > 0 ? n_iters : gpuqo_idp_n_iters);

	// reset it for additional runs 
	level_of_dp = 0;
	return out_qt;
}

template QueryTree<Bitmapset32> *gpuqo_run_dpdp_union<Bitmapset32>(int,  GpuqoPlannerInfo<Bitmapset32>*,int);
template QueryTree<Bitmapset64> *gpuqo_run_dpdp_union<Bitmapset64>(int,  GpuqoPlannerInfo<Bitmapset64>*,int);
template QueryTree<BitmapsetDynamic> *gpuqo_run_dpdp_union<BitmapsetDynamic>(int,  GpuqoPlannerInfo<BitmapsetDynamic>*,int);

