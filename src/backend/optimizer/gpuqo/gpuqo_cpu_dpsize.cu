/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpsize.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsize.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"

typedef std::vector< std::list< std::pair<RelationID, JoinRelation*> > > memo_t;
typedef std::unordered_map<RelationID, JoinRelation*> map_t;

void buildQueryTree(RelationID relid, map_t &joinrels, QueryTree **qt)
{
    auto relid_jr_pair = joinrels.find(relid);
    if (relid_jr_pair == joinrels.end()){
        (*qt) = NULL;
        return;
    }
    JoinRelation* jr = relid_jr_pair->second;

    (*qt) = new QueryTree;
    (*qt)->id = relid;
    (*qt)->left = NULL;
    (*qt)->right = NULL;
    (*qt)->rows = jr->rows;
    (*qt)->cost = jr->cost;

    buildQueryTree(jr->left_relation_id, joinrels, &((*qt)->left));
    buildQueryTree(jr->right_relation_id, joinrels, &((*qt)->right));
}

/* gpuqo_cpu_dpsize
 *
 *	 Sequential CPU baseline for GPU query optimization using the DP size
 *    algorithm.
 */
extern "C"
QueryTree*
gpuqo_cpu_dpsize(BaseRelation baserels[], int N, EdgeInfo edge_table[])
{
    memo_t memo(N+1);
    map_t joinrels;
    QueryTree* out = NULL;

    for(int i=0; i<N; i++){
        JoinRelation *jr = new JoinRelation;
        jr->left_relation_id = 0; 
        jr->right_relation_id = 0; 
        jr->cost = 0.2*baserels[i].rows; 
        jr->rows = baserels[i].rows; 
        jr->edges = baserels[i].edges;
        memo[1].push_back(std::make_pair(baserels[i].id, jr));
        joinrels.insert(std::make_pair(baserels[i].id, jr));
    }

    for (int join_s=2; join_s<=N; join_s++){
        for (int big_s = join_s-1; big_s >= (join_s+1)/2; big_s--){
            int small_s = join_s-big_s;
            for (auto big_i = memo[big_s].begin(); 
                    big_i != memo[big_s].end(); ++big_i){
                for (auto small_i = memo[small_s].begin(); 
                        small_i != memo[small_s].end(); ++small_i){
                    RelationID left_id = big_i->first;
                    JoinRelation *left_rel = big_i->second;
                    RelationID right_id = small_i->first;
                    JoinRelation *right_rel = small_i->second;
                    
                    if (!is_disjoint(left_id, right_id) 
                            || !is_connected(left_id, *left_rel, 
                                             right_id, *right_rel,
                                             baserels, edge_table, N)){

                        continue;
                    }

                    for (int j=0; j<2; j++){
                        RelationID join_id = left_id | right_id;
                        JoinRelation* join_rel = new JoinRelation;

                        join_rel->left_relation_id = left_id;
                        join_rel->right_relation_id = right_id;
                        join_rel->edges = left_rel->edges | right_rel -> edges;

                        join_rel->rows = estimate_join_rows(
                            *join_rel,
                            left_id, *left_rel,
                            right_id, *right_rel,
                            baserels, edge_table, N
                        );
                        join_rel->cost = compute_join_cost(
                            *join_rel,
                            left_id, *left_rel,
                            right_id, *right_rel,
                            baserels, edge_table, N
                        );

                        auto find_iter = joinrels.find(join_id);
                        if (find_iter != joinrels.end()){
                            JoinRelation* old_jr = find_iter->second;
                            if (join_rel->cost < old_jr->cost){
                                *old_jr = *join_rel;
                            }
                            delete join_rel;
                        } else{
                            joinrels.insert(std::make_pair(join_id, join_rel));
                            memo[join_s].push_back(std::make_pair(join_id, join_rel));
                        }

                        // swap and try the opposite
                        right_id = big_i->first;
                        right_rel = big_i->second;
                        left_id = small_i->first;
                        left_rel = small_i->second;
                    }
                    
                }
            } 
        }
    }

    buildQueryTree(memo[N].front().first, joinrels, &out);

    return out;
}


