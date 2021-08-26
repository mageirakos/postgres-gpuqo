#include <iostream>
#include <vector>
#include <unordered_map>
#include "gpuqo.cuh"
#include "gpuqo_query_tree.cuh"

// https://www.techiedelight.com/disjoint-set-data-structure-union-find-algorithm/


// A class to represent a disjoint set
template<typename BitmapsetN>
class DisjointSet
{
//TODO: ama exw node.id BitmapsetN tou base_rel.id tote prepei edw na einai <BitmapsetN, BitmapsetN> kai sto size <BitmapsetN, int>
    std::unordered_map<BitmapsetN, BitmapsetN> parent;
//TODO : borei na thelei na einai public depending sto implementation we'll see
    std::unordered_map<BitmapsetN, int> size;

//TODO : borei na thelei na einai public depending sto implementation we'll see
//TODO: thelw kapoia methods (private logika) wste otan ginetai to Union na ginetai updates kai to csg edw pera
    std::unordered_map<BitmapsetN, BitmapsetN> csg;

public:

    BitmapsetN getCsg(BitmapsetN vertex){
        return csg[Find(vertex)];
    }
    
//TODO: this can be csg.size() so I guess no prob or smt? We will see as I implement csg
    int getSize(BitmapsetN vertex){
        // return size of union in which vertex belongs in
        return size[Find(vertex)];
    }

    void makeSet(GpuqoPlannerInfo<BitmapsetN>* info) 
    {
        // create `n` disjoint sets (one for each item)
        for (int base_rel_idx=0; i < info->n_rels; base_rel_idx++){
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
            csg[x] |= csg[y]
            
        }
        else if (size[x] < size[y]) {
            parent[x] = y;
            size[y] += size[x];
//TODO: Test csg
            csg[y] |= csg[x]

        }
        else {
            parent[x] = y;
            size[x] *= 2;
//TODO: Test csg
            csg[x] |= csg[y]
        }
    }
};

template<typename BitmapsetN>
void printSets(std::vector<int> const &universe, DisjointSet<BitmapsetN> &ds)
{
    for (int i: universe) {
        std::cout << ds.Find(i) << ' ';
        std::cout << '(' << ds.getSize(i) << ')' << '-';
        std::cout << '(' << ds.getCsg(i) << ')' << '-';
    }
    std::cout << std::endl;
}
 

// int main()
// {

// //TODO: auto to universe prepei na einai ta nodes sto graph mou
//     std::vector<int> universe = { 1, 2, 3, 4, 5 };
 
//     // initialize `DisjointSet` class
// // TODO: edw to BitmapsetN den to exw orisei akoma alla tha to orisw otan einai gia to kanoniko compile na ginei
//     DisjointSet<BitmapsetN> ds;
 
//     // create a singleton set for each element of the universe
//     ds.makeSet(universe);
//     printSets(universe, ds);
 
// //TODO: thelw ena logic behind pote kanw Unions (kai pote stamataw na kanw Unions (ena while{})
//     ds.Union(4, 3);        // 4 and 3 are in the same set
//     printSets(universe, ds);
 
//     ds.Union(2, 1);        // 1 and 2 are in the same set
//     printSets(universe, ds);
 
//     ds.Union(1, 3);        // 1, 2, 3, 4 are in the same set
//     printSets(universe, ds);
 
//     return 0;
// }
// #################################################################################################################################################################################

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
QueryTree<BitmapsetOuter> *gpuqo_run_dpdp_union_dp(int gpuqo_algo, 
						GpuqoPlannerInfo<BitmapsetOuter>* info,
						list<remapper_transf_el_t<BitmapsetOuter> > &remap_list) 
{
	printf("\n f(x): gpuqo_run_dpdp_union_dp for LEVEL %d\n", level_of_dp);

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
QueryTree<BitmapsetOuter> *gpuqo_run_dpdp_union_rec(int gpuqo_algo, 
					GpuqoPlannerInfo<BitmapsetOuter>* info,
					list<remapper_transf_el_t<BitmapsetOuter> > &remap_list,
					int n_iters) 
{
	level_of_dp++;
	std::cout << "\n\t LEVEL OF DP: " << level_of_dp << std::endl;
	printf("\n f(x): gpuqo_run_dpdp_union_rec ----- FIRST CHECK ----- LEVEL %d \n", level_of_dp);
	Remapper<BitmapsetOuter, BitmapsetInner> remapper(remap_list);
	GpuqoPlannerInfo<BitmapsetInner> *new_info = remapper.remapPlannerInfo(info);
	
	new_info->n_iters = min(new_info->n_rels, n_iters);

	if (new_info->n_rels == new_info->n_iters){ // its going to be equal because of the above min()
		printf("\n f(x): gpuqo_run_dpdp_union_rec => -----INSIDE TERMINATION CHECK ----- LEVEL %d \n", level_of_dp);
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
		printf("\n f(x): gpuqo_run_dpdp_union_rec => returning -----INSIDE TERMINATION CHECK -----\n");
		return out_qt;
	}



	// std::cout << "BEFORE gpuqo_k_cut_edges = " << gpuqo_k_cut_edges << " new_info->n_rels= " << new_info->n_rels << std::endl;
	// if (gpuqo_k_cut_edges >= new_info->n_rels-1){ // -1 because infinite loop since we won't decrease at all
	// 	gpuqo_k_cut_edges = new_info->n_rels - 20; // -20 so that graph decreases by at least 10 node each time
	// } 
	// std::cout << "AFTER gpuqo_k_cut_edges = " << gpuqo_k_cut_edges << " new_info->n_rels= " << new_info->n_rels << std::endl;


// FIND EDGES TO CUT (NOT NEEDED)

	// std::vector<GraphEdge<BitmapsetInner>> cutEdges = find_k_cut_edges(new_info, gpuqo_k_cut_edges);
	// printf("FOUND %zu cutEdges", cutEdges.size());
	// printf("\tPRINTING  CUT EDGES\n, where 1st rel(2^1) = 1 (start counting from 1 in rel id)");
	// for(int i=0; i<gpuqo_k_cut_edges; i++){
	// 	std::cout << "edge_" << i << " : left= " << cutEdges[i].left.toUlonglong() << "\tright= " << cutEdges[i].right.toUlonglong() \
	// 	<<  "\t(" << cutEdges[i].left.lowestPos() << "-" << cutEdges[i].right.lowestPos()-1 << ") " << std::endl;		
	// }

// CUT THE EDGES (NOT NEEDED)
	// printf("\n f(x): gpuqo_run_dpdp_union_rec => after find_k_cut_edges ----- CHECK 2 ----- \n");
	// BitmapsetInner edge_table_copy[BitmapsetInner::SIZE];
	// printf("\n edge table BEFORE: \n");
	// printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	// std::cout << "                 3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	// for (int i=0; i < new_info->n_rels; i++){
	// 	// std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i].toUlonglong() << std::endl;
	// 	std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i] << std::endl;
	// 	edge_table_copy[i] = new_info->edge_table[i];
	// }

	// for (int i=0; i < gpuqo_k_cut_edges; i++){  
	// 	GraphEdge<BitmapsetInner> cut_edge = cutEdges[i];
	// 	std::cout << "removing edge: " << cut_edge.left.lowestPos() << "--" << cut_edge.right.lowestPos() << " - 1st rel(2^1) = 1" << std::endl;
	// 	edge_table_copy[cut_edge.left.lowestPos()-1].unset(cut_edge.right.lowestPos()); 
	// 	edge_table_copy[cut_edge.right.lowestPos()-1].unset(cut_edge.left.lowestPos()); 
	// }
	
	// Lopp only for printing edge_table AFTER
	// printf("\n edge table AFTER: \n");
	// printf("\n BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	// for (int i=0; i < new_info->n_rels; i++){
	// 	std::cout << "edge_table["<< i <<"] = "<< new_info->edge_table[i] << std::endl;
	// }
	
	// printf("\nsubset_baserel_id BitmapsetInner::SIZE = %d\n", BitmapsetInner::SIZE);
	// BitmapsetInner subset_baserel_id = BitmapsetInner(0);
	// for (int i = 0; i < new_info->n_rels; i++){
	// 	std::cout << "iter: " << i << " new_info->base_rels[" << i << "].id = " << new_info->base_rels[i].id.toUlonglong() << std::endl;
	// 	subset_baserel_id |= new_info->base_rels[i].id;
	// }
	// std::cout << "                  3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	// std::cout << "subset_baserel_id " << subset_baserel_id << std::endl;

	

// CREATE SUBGRAPHS (WHICH SHOULD BE DONE WITH THE UNION FIND NOT NEEDED AS IS HERE)
	// printf("\nPRINTING SUBGRAPHS: \n");
	// std::vector<BitmapsetInner> subgraphs;
	// std::cout << " 3210987654321098765432109876543210987654321098765432109876543210" << std::endl; // index up to 64
	// while(subset_baserel_id!=0){
	// 	BitmapsetInner csg = grow(subset_baserel_id.lowest(), subset_baserel_id, edge_table_copy);
	// 	std::cout << "before : " << subset_baserel_id.toUlonglong() << " from csg: " << csg.toUlonglong() << std::endl; 
	// 	subset_baserel_id = subset_baserel_id.differenceSet(csg);
	// 	std::cout << "after : " << subset_baserel_id.toUlonglong() << " from csg: " << csg.toUlonglong() << std::endl; 
	// 	subgraphs.push_back(csg);
	// 	std::cout << "csg_" << subgraphs.size() << " : " << csg.toUlonglong() <<std::endl; 
	// 	std::cout << subset_baserel_id << std::endl;
	// }


//TODO: Create the subgraphs (csgs) with Union Find dataset based on some starting/stopping logic around sizes
	std::vector<BitmapsetInner> subgraphs;

// 1. BFS gia na kaneis assign weights kai na kaneis create tha LeafNodes kai initialize to LeafPriorityQueue

// NOTE: To UnionPriorityQueue exei to union_id, node_id, size, edge_weight (opou edge weight) to minimum edge
//?  apo ola ta edges tou node_id? 
//? Pws kanw keep track of this thing?

//? What if UnionQueue is priority queueu of the edges? ( theloume omws to minimum size na kanei grow prwta)
// ara prepei 
// NOMIZE ETSI PREPEI NA GINEI
// ME KWDIKA OMWS PWS THA TO KANW

template<typename BitmapsetN>
struct UnionEdge{
	BitmapsetN left;
	BitmapsetN right;
	int left_size; //TODO: This needs to be created with ds.getSize(node_id) of DisjointSet or else it will be wrong
	int right_size;

// What if priority queue of total_size and edge_weight??
// Etsi gemizoume to queueu me OLA ta edges
// kathe fora pou kanoume POP elegxoume an left + right sto idio UNION kai continue (pop again)
// sorted by total size + spaei to equivilancy me edge_weight?    
    int total_size = left_size + right_size; 
    float edge_weight; // minimum weight apo ta edges tou node_id
};


// 1. Create Union our of all nodes 
// 2. Start with leaf nodes and try to grow (union)
    // - This should be done with some sort of BFS traversal?
    // - So that as we traverse we do some check on if we should union and then union?
    // - Stop growing union (csg) after size is around 20-25
// 3. Go to another leaf and start growing from that?



// ? An kanw to priority queue pou kanei rank kala starting points pou a3izei na kanoume twra grow from
// tha prepei na kanw ena arxiko BFS, na dwsw weights, na vrw leaves, na krataw ranks twn nodes etc.
// Episis kathws megalwnei to union twn nodes apo ta priority queue kapws na to 3anavazw sto priority queue?
// i den xreiazetai?


// THIS SHOULD BE THE SAME AFTER WE CREATE "SUBGRAPHS" WITH UNION-FIND
	printf("\n f(x): gpuqo_run_dpdp_union_rec => after subgraphs ----- CHECK 3 ----- \n");
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


	printf("\n f(x): gpuqo_run_dpdp_union_rec => recursing ----- CHECK 4 ----- LEVEL %d \n", level_of_dp);
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
	printf("\n f(x): gpuqo_run_dpdp_union_rec => after recursion ----- CHECK 5 ----- LEVEL %d \n", level_of_dp);
	QueryTree<BitmapsetOuter> *out_qt = remapper.remapQueryTree(res_qt);
	freeGpuqoPlannerInfo(new_info);
	freeQueryTree(res_qt);
	printf("\n f(x): gpuqo_run_dpdp_union_rec => returning ----- FINAL CHECK ----- LEVEL %d \n", level_of_dp);
	return out_qt;
}


template<typename BitmapsetN>
QueryTree<BitmapsetN> *gpuqo_run_dpdp_union(int gpuqo_algo, 
									GpuqoPlannerInfo<BitmapsetN>* info,
									int n_iters)
{
	printf("\n f(x): gpu_run_dpdp_union \n");
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

