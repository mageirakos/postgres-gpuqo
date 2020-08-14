/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpe.cu
 *      Generic implementation of the DPE CPU algorithm.
 *
 * src/backend/optimizer/gpuqo/gpuqo_cpu_dpe.cu
 *
 *-------------------------------------------------------------------------
 */

#include <list>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <cstdint>
#include "pthread.h"
#include <stdio.h>
#include <errno.h>

#include "optimizer/gpuqo_common.h"

#include "optimizer/gpuqo.cuh"
#include "optimizer/gpuqo_timing.cuh"
#include "optimizer/gpuqo_debug.cuh"
#include "optimizer/gpuqo_cost.cuh"
#include "optimizer/gpuqo_filter.cuh"
#include "optimizer/gpuqo_cpu_common.cuh"
#include "optimizer/gpuqo_cpu_dpe.cuh"
#include "optimizer/gpuqo_dependency_buffer.cuh"

// set by GUC
int gpuqo_dpe_n_threads;
int gpuqo_dpe_pairs_per_depbuf;

typedef struct DependencyBuffers{
    DependencyBuffer* depbuf_curr;
    DependencyBuffer* depbuf_next;
    pthread_cond_t avail_jobs;
    pthread_mutex_t depbuf_mutex;
    bool finish;
} DependencyBuffers;

typedef struct ThreadArgs{
    int id;
    BaseRelation *base_rels;
    int n_rels;
    EdgeInfo *edge_table;
    memo_t *memo;
    DependencyBuffers* depbufs;
} ThreadArgs;

typedef struct DPEExtra{
    int job_count;
    pthread_t* threads;
    DependencyBuffers depbufs;    
    ThreadArgs* thread_args;
} DPEExtra;

void process_depbuf(DependencyBuffer* depbuf, BaseRelation *base_rels, 
                    int n_rels, EdgeInfo *edge_table){
    depbuf_entry_t job;
    while ((job = depbuf->pop()).first != NULL){
        JoinRelationDPE *memo_join_rel = job.first;
        for (auto iter = job.second.begin(); iter != job.second.end(); ++iter){
            JoinRelationDPE *left_rel = iter->first;
            JoinRelationDPE *right_rel = iter->second;

            while (left_rel->num_entry != 0) 
                ; // busy wait for entries to be ready
            while (right_rel->num_entry != 0)
                ; // busy wait for entries to be ready

            JoinRelationDPE* join_rel = make_join_relation(
                *left_rel, *right_rel, base_rels, n_rels, edge_table
            );
            
            if (join_rel->cost < memo_join_rel->cost){
                // copy only the JoinRelation part, not num_entry
                *((JoinRelation*)memo_join_rel) = *((JoinRelation*)join_rel);
            }

            delete join_rel;
        }
    }
}

void wait_and_swap_depbuf(DPEExtra* extra, BaseRelation *base_rels, 
                            int n_rels, EdgeInfo *edge_table){
    // lend an hand to worker threads
    process_depbuf(extra->depbufs.depbuf_curr, base_rels, n_rels, edge_table);

    // swap depbufs
    pthread_mutex_lock(&extra->depbufs.depbuf_mutex);

    depbuf_temp = extra->targs.depbuf_curr;

    // swap depbufs
    DependencyBuffer* depbuf_temp = extra->depbufs.depbuf_curr;
    extra->depbufs.depbuf_curr = extra->depbufs.depbuf_next;
    extra->depbufs.depbuf_next = depbuf_temp;

    // signal threads that they can start executing
    pthread_cond_broadcast(&extra->depbufs.avail_jobs);

    pthread_mutex_unlock(&extra->depbufs.depbuf_mutex);
}

bool submit_join(int level, JoinRelationDPE* &join_rel, 
            JoinRelationDPE &left_rel, JoinRelationDPE &right_rel,
            BaseRelation* base_rels, int n_rels, EdgeInfo* edge_table, 
            memo_t &memo, extra_t extra){
    bool out;
    RelationID relid = BMS64_UNION(left_rel.id, right_rel.id);

    auto find_iter = memo.find(relid);
    if (find_iter != memo.end()){
        join_rel = (JoinRelationDPE*) find_iter->second;
        out = false;
    } else{
        join_rel = build_join_relation<JoinRelationDPE>(left_rel, right_rel);
        join_rel->num_entry = 0;
        memo.insert(std::make_pair(join_rel->id, join_rel));
        out = true;
    }

    DPEExtra* mExtra = (DPEExtra*) extra.impl;

    if (mExtra->job_count < gpuqo_dpe_pairs_per_depbuf){
        mExtra->depbufs.depbuf_next->push(join_rel, &left_rel, &right_rel);
        mExtra->job_count++;
    } else {
        wait_and_swap_depbuf(mExtra, base_rels, n_rels, edge_table);
        mExtra->job_count = 0;
    }

    return out;
}

// instead of join it is more of a distribution of jobs
void gpuqo_cpu_dpe_join(int level, bool try_swap,
                            JoinRelation &left_rel, JoinRelation &right_rel,
                            BaseRelation* base_rels, int n_rels, 
                            EdgeInfo* edge_table, memo_t &memo, extra_t extra, 
                            struct DPCPUAlgorithm algorithm){
    if (algorithm.check_join_function(level, left_rel, right_rel,
                            base_rels, n_rels, edge_table, memo, extra)){
        JoinRelationDPE *join_rel1, *join_rel2;
        bool new_joinrel;
        new_joinrel = submit_join(level, join_rel1, 
                (JoinRelationDPE&) left_rel, (JoinRelationDPE&) right_rel, 
                base_rels, n_rels, edge_table, 
                memo, extra
        );
        algorithm.post_join_function(level, new_joinrel, 
                            *((JoinRelation*)join_rel1), 
                            left_rel,  right_rel, base_rels, n_rels,
                            edge_table, memo, extra);
        if (try_swap){
            new_joinrel = submit_join(level, join_rel2, 
                (JoinRelationDPE&) left_rel, (JoinRelationDPE&) right_rel, 
                base_rels, n_rels, edge_table, 
                memo, extra
            );
            algorithm.post_join_function(level, new_joinrel, 
                                *((JoinRelation*)join_rel2), 
                                left_rel, right_rel, base_rels, n_rels,
                                edge_table, memo, extra);
        }
    }
}

void* thread_function(void* _args){
    ThreadArgs *args = (ThreadArgs*) _args;

    while(true){
        pthread_mutex_lock(&args->depbufs->depbuf_mutex);
        while(args->depbufs->depbuf_curr->empty() 
                && !args->depbufs->finish){
            pthread_cond_wait(&args->depbufs->avail_jobs, 
                            &args->depbufs->depbuf_mutex);
        }
        pthread_mutex_unlock(&args->depbufs->depbuf_mutex);

        if (args->depbufs->finish)
            return NULL;

        process_depbuf(args->depbufs->depbuf_curr, args->base_rels, 
            args->n_rels, args->edge_table);
    }

    return NULL;
}

QueryTree* gpuqo_cpu_dpe(BaseRelation base_rels[], int n_rels, 
                             EdgeInfo edge_table[], DPCPUAlgorithm algorithm){
    
    DECLARE_TIMING(gpuqo_cpu_dpe);
    START_TIMING(gpuqo_cpu_dpe);

    extra_t extra;
    memo_t memo;
    QueryTree* out = NULL;

    extra.impl = (void*) new DPEExtra;
    DPEExtra* mExtra = (DPEExtra*) extra.impl;

    mExtra->threads = new pthread_t[gpuqo_dpe_n_threads-1];
    mExtra->thread_args = new ThreadArgs[gpuqo_dpe_n_threads-1];
    mExtra->job_count = 0;
    
    mExtra->depbufs.finish = false;
    mExtra->depbufs.depbuf_curr = new DependencyBuffer(n_rels);
    mExtra->depbufs.depbuf_next = new DependencyBuffer(n_rels);
    pthread_cond_init(&mExtra->depbufs.avail_jobs, NULL);
    pthread_mutex_init(&mExtra->depbufs.depbuf_mutex, NULL);

    for (int i=0; i<gpuqo_dpe_n_threads-1; i++){
        mExtra->thread_args[i].id = i;
        mExtra->thread_args[i].base_rels = base_rels;
        mExtra->thread_args[i].n_rels = n_rels;
        mExtra->thread_args[i].edge_table = edge_table;
        mExtra->thread_args[i].memo = &memo;
        mExtra->thread_args[i].depbufs = &mExtra->depbufs;
        
        int ret = pthread_create(&mExtra->threads[i], NULL, thread_function, 
                                (void*) &mExtra->thread_args[i]);

        if (ret != 0){
            perror("pthread_create: ");
            return NULL;
        }
    }

    for(int i=0; i<n_rels; i++){
        JoinRelationDPE *jr = new JoinRelationDPE;
        jr->id = base_rels[i].id; 
        jr->left_relation_id = 0; 
        jr->left_relation_ptr = NULL; 
        jr->right_relation_id = 0; 
        jr->right_relation_ptr = NULL; 
        jr->cost = 0.2*base_rels[i].rows; 
        jr->rows = base_rels[i].rows; 
        jr->edges = base_rels[i].edges;
        jr->num_entry = 0;
        memo.insert(std::make_pair(base_rels[i].id, (JoinRelation*) jr));
    }

    algorithm.init_function(base_rels, n_rels, edge_table, memo, extra);
    
    algorithm.enumerate_function(base_rels, n_rels, edge_table,gpuqo_cpu_dpe_join, memo, extra, algorithm);

    // finish depbuf_curr and set depbuf_next
    wait_and_swap_depbuf(mExtra, base_rels, n_rels, edge_table);
    // help finishing depbuf_next (which is now in depbuf_curr)
    process_depbuf(mExtra->depbufs.depbuf_curr, base_rels, n_rels, edge_table);

    // stop worker threads
    pthread_mutex_lock(&mExtra->depbufs.depbuf_mutex);
    mExtra->depbufs.finish = true;

    // awake threads to let them realize it's over
    pthread_cond_broadcast(&mExtra->depbufs.avail_jobs);

    pthread_mutex_unlock(&mExtra->depbufs.depbuf_mutex);

    // wait threads to exit
    for (int i = 0; i < gpuqo_dpe_n_threads-1; i++){
        pthread_join(mExtra->threads[i], NULL);
    }

    RelationID final_joinrel_id = 0ULL;
    for (int i = 0; i < n_rels; i++)
        final_joinrel_id = BMS64_UNION(final_joinrel_id, base_rels[i].id);

    
    auto final_joinrel_pair = memo.find(final_joinrel_id);
    if (final_joinrel_pair != memo.end())
        build_query_tree(final_joinrel_pair->second, memo, &out);

    // delete all dynamically allocated memory
    for (auto iter=memo.begin(); iter != memo.end(); ++iter){
        delete iter->second;
    }

    algorithm.teardown_function(base_rels, n_rels, edge_table, memo, extra);
    
    pthread_cond_destroy(&mExtra->depbufs.avail_jobs);
    pthread_mutex_destroy(&mExtra->depbufs.depbuf_mutex);
    delete mExtra->threads;
    delete mExtra->thread_args;
    delete mExtra;


    STOP_TIMING(gpuqo_cpu_dpe);
    PRINT_TIMING(gpuqo_cpu_dpe);

    return out;
}