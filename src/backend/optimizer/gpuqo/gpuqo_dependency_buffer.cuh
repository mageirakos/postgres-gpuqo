/*-------------------------------------------------------------------------
 *
 * gpuqo_dependency_buffer.cuh
 *	  declaration of the DependencyBuffer class used by the DPE algorithm
 *
 * src/include/optimizer/gpuqo_dependency_buffer.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_DEPENDENCY_BUFFER_CUH
#define GPUQO_DEPENDENCY_BUFFER_CUH

#include <deque>
#include <unordered_map>
#include <pthread.h>
#include <atomic>

#include "gpuqo.cuh"
#include "gpuqo_cpu_common.cuh"

template <typename BitmapsetN>
struct JoinRelationDPE : public JoinRelationCPU<BitmapsetN> {
    std::atomic_int num_entry;
};

template<typename BitmapsetN>
struct DepBufNode{
    JoinRelationDPE<BitmapsetN> *join_rel;
    JoinRelationDPE<BitmapsetN> *left_rel;
    JoinRelationDPE<BitmapsetN> *right_rel;
    DepBufNode<BitmapsetN>* next_join;
    DepBufNode<BitmapsetN>* prev_join;
    DepBufNode<BitmapsetN>* next_rel;
    DepBufNode<BitmapsetN>* prev_rel;
};

template<typename BitmapsetN>
class DependencyBuffer{   
private:
    int n_rels;
    int capacity;
    DepBufNode<BitmapsetN>** queues;
    std::atomic<DepBufNode<BitmapsetN>*> unified_queue;
    DepBufNode<BitmapsetN>* free_nodes;
    DepBufNode<BitmapsetN>* next_free_node;
public:
    DependencyBuffer(int n_rels, int capacity);
    void push(
        JoinRelationDPE<BitmapsetN> *join_rel, 
        JoinRelationDPE<BitmapsetN> *left_rel, 
        JoinRelationDPE<BitmapsetN> *right_rel);
    DepBufNode<BitmapsetN> *pop();
    bool empty();
    bool full();
    void clear();
    void unify_queues();
    ~DependencyBuffer();
};
	
#endif							/* GPUQO_DEPENDENCY_BUFFER_CUH */
