/*------------------------------------------------------------------------
 *
 * gpuqo_dpsub_csg.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_csg.cu
 *
 *-------------------------------------------------------------------------
 */

#include <iostream>
#include <cmath>
#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/system/system_error.h>
#include <thrust/distance.h>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_dpsub.cuh"

// user-defined variables
bool gpuqo_dpsub_csg_enable;
int gpuqo_dpsub_csg_threshold;

typedef thrust::tuple<RelationID, RelationID> emit_stack_elem_t;
typedef thrust::tuple<RelationID, RelationID, RelationID, RelationID> loop_stack_elem_t;

template<int MAX_DEPTH>
__device__
void enumerate_sub_csg(RelationID T, RelationID I, RelationID E,
                    JoinRelation &jr_out, JoinRelation* memo_vals,
                    BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table){
    // S, X
    emit_stack_elem_t emit_stack[MAX_DEPTH];
    size_t emit_stack_size = 0;
    // S, subset, X, N
    loop_stack_elem_t loop_stack[MAX_DEPTH];
    size_t loop_stack_size = 0;

    RelationID temp;
    if (I != BMS64_EMPTY){
        temp = BMS64_LOWEST(I);
    } else{
        temp = BMS64_DIFFERENCE(T, E);
    }

    while (temp != BMS64_EMPTY){
        int idx = BMS64_HIGHEST_POS(temp)-1;
        RelationID v = BMS64_NTH(idx);
        
        emit_stack[emit_stack_size++] = thrust::make_tuple(v,
                                            BMS64_SET_ALL_LOWER_INC(v));
        temp = BMS64_DIFFERENCE(temp, v);
    }

    while (emit_stack_size != 0 || loop_stack_size != 0){
        if (emit_stack_size != 0){
            emit_stack_elem_t top = emit_stack[--emit_stack_size];
            RelationID S = top.get<0>();
            RelationID X = top.get<1>();

#ifdef GPUQO_DEBUG
            printf("[%llu: %llu, %llu] emit_stack: S=%llu, X=%llu\n", T, I, E, S, X);
#endif
            
            if (BMS64_IS_SUBSET(I, S)){
                try_join(T, jr_out, S, BMS64_DIFFERENCE(T, S), 
                    memo_vals, base_rels, n_rels, edge_table);
            }

            RelationID N = BMS64_INTERSECTION(
                BMS64_DIFFERENCE(get_neighbours(S, base_rels, n_rels), X),
                BMS64_DIFFERENCE(T, E)
            );
            RelationID lowI = BMS64_LOWEST(BMS64_DIFFERENCE(I, S));
        
            // If possible, directly move to smaller I (it does not make 
            // sense to explore other rels in I first since it won't be 
            // possible to go back)
            if (BMS64_INTERSECTS(lowI,N))
                N = lowI;

            loop_stack[loop_stack_size++] = thrust::make_tuple(
                S, BMS64_EMPTY, X, N
            );
        }

        if (loop_stack_size != 0){
            loop_stack_elem_t top = loop_stack[--loop_stack_size];
            RelationID S = top.get<0>();
            RelationID subset = top.get<1>();
            RelationID X = top.get<2>();
            RelationID N = top.get<3>();

#ifdef GPUQO_DEBUG
            printf("[%llu: %llu, %llu] loop_stack: S=%llu, subset=%llu, X=%llu, N=%llu\n", T, I, E, S, subset, X, N);
#endif

            subset = BMS64_NEXT_SUBSET(subset, N);
            if (subset != BMS64_EMPTY){
                loop_stack[loop_stack_size++] = thrust::make_tuple(
                    S, subset, X, N
                );
                emit_stack[emit_stack_size++] = thrust::make_tuple(
                    BMS64_UNION(S,subset), BMS64_UNION(X, N)
                );
            }
        }
    }
}

__device__
JoinRelation dpsubEnumerateCsg::operator()(RelationID relid, uint64_t cid)
{
    uint64_t qss = BMS64_SIZE(relid);
    uint64_t splits_per_qs = ceil_div((1<<qss), n_pairs);
    
    uint64_t dpccp_splits = BMS64_HIGHEST(splits_per_qs);
    uint64_t cmp_cid = dpccp_splits-1 - cid;

    JoinRelation jr_out;
    jr_out.id = BMS64_EMPTY;
    jr_out.cost = INFD;

    if (cid < dpccp_splits){
#ifdef GPUQO_DEBUG
        printf("[%llu, %llu] splits_per_qs=%llu, dpccp_splits=%llu, cmp_cid=%llu\n", 
            relid, cid, splits_per_qs, dpccp_splits, cmp_cid);
#endif
        Assert(BMS64_UNION(cid, cmp_cid) == dpccp_splits-1);
        Assert(!BMS64_INTERSECTS(cid, cmp_cid));

        RelationID inc_set = BMS64_EXPAND_TO_MASK(cid, relid);
        RelationID exc_set = BMS64_EXPAND_TO_MASK(cmp_cid, relid);
        
        enumerate_sub_csg<64>(relid, inc_set, exc_set, jr_out, 
                memo_vals.get(), base_rels.get(), sq, edge_table.get());
    }
    
    return jr_out;
}
