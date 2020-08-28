/*-------------------------------------------------------------------------
 *
 * gpuqo_dpsub.cuh
 *	  declaration of gpuqo DPsub related functions and functors.
 *
 * src/include/optimizer/gpuqo_dpsub.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DPSUB_CUH
#define GPUQO_DPSUB_CUH

#include <thrust/tabulate.h>

__device__
RelationID dpsub_unrank_sid(uint64_t sid, uint64_t qss, uint64_t sq, uint64_t* binoms);

__device__
void try_join(RelationID relid, JoinRelation &jr_out, 
            RelationID l, RelationID r, JoinRelation* memo_vals,
            BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table);

/* unrankEvaluateDPSub
 *
 *	 unrank algorithm for DPsub GPU variant with embedded evaluation and 
 *   partial pruning.
 */
struct unrankEvaluateDPSub : public thrust::unary_function< uint64_t,thrust::tuple<RelationID, JoinRelation> >
{
    thrust::device_ptr<JoinRelation> memo_vals;
    thrust::device_ptr<BaseRelation> base_rels;
    thrust::device_ptr<EdgeInfo> edge_table;
    thrust::device_ptr<uint64_t> binoms;
    int sq;
    int qss;
    uint64_t offset;
    int n_pairs;
public:
    unrankEvaluateDPSub(
        thrust::device_ptr<JoinRelation> _memo_vals,
        thrust::device_ptr<BaseRelation> _base_rels,
        int _sq,
        thrust::device_ptr<EdgeInfo> _edge_table,
        thrust::device_ptr<uint64_t> _binoms,
        int _qss,
        uint64_t _offset,
        int _n_pairs
    ) : memo_vals(_memo_vals), base_rels(_base_rels), sq(_sq), 
        edge_table(_edge_table), binoms(_binoms), qss(_qss), offset(_offset),
        n_pairs(_n_pairs)
    {}

    __device__
    thrust::tuple<RelationID, JoinRelation> operator()(uint64_t tid);
};

#endif							/* GPUQO_DPSUB_CUH
