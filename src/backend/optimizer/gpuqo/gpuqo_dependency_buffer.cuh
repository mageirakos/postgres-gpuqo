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
using join_list_t = std::deque< std::pair<JoinRelationDPE<BitmapsetN>*, JoinRelationDPE<BitmapsetN>*> >;

template<typename BitmapsetN>
using depbuf_entry_t = std::pair<JoinRelationDPE<BitmapsetN>*, join_list_t<BitmapsetN>*>;

template<typename BitmapsetN>
using depbuf_queue_t = std::deque<depbuf_entry_t<BitmapsetN> >;


template<typename BitmapsetN>
using depbuf_lookup_t = std::unordered_map<BitmapsetN, join_list_t<BitmapsetN>*>;

template<typename BitmapsetN>
class DependencyBuffer{
private:
    int n_rels;
    std::pair<depbuf_queue_t<BitmapsetN>, depbuf_lookup_t<BitmapsetN> > *queue_lookup_pairs;
    pthread_mutex_t mutex;
    int first_non_empty;
public:
    DependencyBuffer(int n_rels);
    void push(
        JoinRelationDPE<BitmapsetN> *join_rel, 
        JoinRelationDPE<BitmapsetN> *left_rel, 
        JoinRelationDPE<BitmapsetN> *right_rel);
    depbuf_entry_t<BitmapsetN> pop();
    bool empty();
    void clear();
    size_t size();
    ~DependencyBuffer();
};
	
#endif							/* GPUQO_DEPENDENCY_BUFFER_CUH */
