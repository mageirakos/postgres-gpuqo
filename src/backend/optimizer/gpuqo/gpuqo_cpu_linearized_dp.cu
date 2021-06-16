/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_linearized_dp.cu
 *      Linearized DP using IKKBZ
 *
 * src/backend/optimizer/gpuqo/gpuqo_cpu_linearized_dp.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"
#include "gpuqo_cpu_common.cuh"
#include "gpuqo_query_tree.cuh"

using namespace std;

template<typename BitmapsetN>
static int
qt_to_lin_remap(QueryTree<BitmapsetN> *qt, 
                        list<remapper_transf_el_t<BitmapsetN> > &remap_list,
                        int idx)
{
    if (qt == NULL) { // nothing
        return 0;
    } else if (qt->left == NULL && qt->right == NULL){ // leaf
        remapper_transf_el_t<BitmapsetN> el;
        el.qt = NULL;
        el.to_idx = idx;
        el.from_relid = qt->id;
        remap_list.push_back(el);
        return 1;
    } else { // any node
        int n_left = qt_to_lin_remap(qt->left, remap_list, idx);
        int n_right = qt_to_lin_remap(qt->right, remap_list, idx+n_left);
        return n_left + n_right;
    }    
}

/* gpuqo_cpu_ikkbz
 *
 *	 IKKBZ approximation
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_linearized_dp(GpuqoPlannerInfo<BitmapsetN> *orig_info)
{
    QueryTree<BitmapsetN> *ikkbz_qt = gpuqo_cpu_ikkbz(orig_info);
    
    list<remapper_transf_el_t<BitmapsetN> > remap_list;
    qt_to_lin_remap(ikkbz_qt, remap_list, 0);
    Remapper<BitmapsetN,BitmapsetN> remapper(remap_list);
    GpuqoPlannerInfo<BitmapsetN> *info = remapper.remapPlannerInfo(orig_info);

    freeQueryTree(ikkbz_qt);

    vector<vector<JoinRelationCPU<BitmapsetN>* > > T;
    T.resize(info->n_rels, 
        vector<JoinRelationCPU<BitmapsetN>* >(info->n_rels, NULL));
    
    for (int i = 0; i < info->n_rels; i++) {
        JoinRelationCPU<BitmapsetN>* bjr = new JoinRelationCPU<BitmapsetN>;
        bjr->id = info->base_rels[i].id; 
        bjr->left_rel_id = 0; 
        bjr->left_rel_ptr = NULL; 
        bjr->right_rel_id = 0; 
        bjr->right_rel_ptr = NULL; 
        bjr->cost = cost_baserel(info->base_rels[i]); 
        bjr->width = info->base_rels[i].width; 
        bjr->rows = info->base_rels[i].rows; 
        bjr->edges = info->edge_table[i];
        T[i][i] = bjr;
    }

    for (int s = 2; s <= info->n_rels; s++){
        for (int i = 0; i <= info->n_rels-s; i++){
            for (int j = 1; j < s; j++){
                JoinRelationCPU<BitmapsetN> *l = T[i][i+j-1];
                JoinRelationCPU<BitmapsetN> *r = T[i+j][i+s-1];

                if (l == NULL || r == NULL)
                    continue;
                
                if (are_connected_ids(l->id, r->id, info)) {
                    JoinRelationCPU<BitmapsetN> *p = make_join_relation(
                                                                *l, *r, info);
                    if (T[i][i+s-1] == NULL 
                            || p->cost.total < T[i][i+s-1]->cost.total) 
                    {
                        T[i][i+s-1] = p;
                    }
                }
            }
        }

    }

    QueryTree<BitmapsetN> *qt;
    build_query_tree(T[0][info->n_rels-1], &qt);
    QueryTree<BitmapsetN> *qt_remap = remapper.remapQueryTree(qt);

    for (int i = 0; i < info->n_rels; i++){
        for (int j = 0; j < info->n_rels; j++){
            if (T[i][j]){
                delete T[i][j];
                T[i][j] = NULL;
            }
        }
    }

    freeGpuqoPlannerInfo(info);
	freeQueryTree(qt);

    return qt_remap;
}

template QueryTree<Bitmapset32>* gpuqo_cpu_linearized_dp<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_linearized_dp<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic>* gpuqo_cpu_linearized_dp<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>*);
