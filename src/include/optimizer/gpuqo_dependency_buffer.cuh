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

#include <list>
#include <unordered_map>
#include <pthread.h>
#include <atomic>

#include "optimizer/gpuqo_common.h"
#include "optimizer/gpuqo.cuh"

struct JoinRelationDPE : JoinRelation{
    std::atomic_int num_entry;
};

typedef std::list< std::pair<JoinRelationDPE*, JoinRelationDPE*> >  depbuf_list_t;
typedef std::pair<JoinRelationDPE*, depbuf_list_t> depbuf_entry_t;
typedef std::list<depbuf_entry_t> depbuf_queue_t;
typedef std::unordered_map<RelationID, depbuf_entry_t*> depbuf_lookup_t;

class DependencyBuffer{
private:
    int n_rels;
    std::pair<depbuf_queue_t, depbuf_lookup_t> *queue_lookup_pairs;
    pthread_mutex_t mutex;
    int first_non_empty;
public:
    DependencyBuffer(int n_rels);
    void push(JoinRelationDPE *join_rel, JoinRelationDPE *left_rel, JoinRelationDPE *right_rel);
    depbuf_entry_t pop();
    bool empty();
    void clear();
    ~DependencyBuffer();
};
	
#endif							/* GPUQO_DEPENDENCY_BUFFER_CUH */
