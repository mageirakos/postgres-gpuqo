/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_dpsub_bicc_parallel.cu
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpsub_parallel.cu
 *
 *-------------------------------------------------------------------------
 */

 #include <list>
 #include <vector>
 #include <stack>
 #include <iostream>
 #include <cmath>
 #include <cstdint>
 
 #include "optimizer/gpuqo_common.h"
 
 #include "gpuqo.cuh"
 #include "gpuqo_timing.cuh"
 #include "gpuqo_debug.cuh"
 #include "gpuqo_cost.cuh"
 #include "gpuqo_filter.cuh"
 #include "gpuqo_cpu_dpsub.cuh"
 #include "gpuqo_dpsub.cuh"
 #include "gpuqo_binomial.cuh"

using namespace std;

int gpuqo_cpu_dpsub_parallel_chunk_size;

template<typename BitmapsetN>
struct ThreadArgs{
    int id;
    alg_t<BitmapsetN>* alg;
    level_hashtable<BitmapsetN>* memo;
    pthread_cond_t* thread_waiting_cond;
    pthread_cond_t* start_thread_cond;
    pthread_mutex_t* mutex;
    int* n_waiting;
    int* level;
    uint_t<BitmapsetN>* skip;    
    atomic<uint_t<BitmapsetN>>* next_sid;
    uint_t<BitmapsetN> *binoms;
    GpuqoPlannerInfo<BitmapsetN> *info;
};

template<typename BitmapsetN>
static void* thread_function(void* _args){
    ThreadArgs<BitmapsetN> *args = (ThreadArgs<BitmapsetN>*) _args;
    int level = 0;

    while(level < args->info->n_iters){
        list<pair<BitmapsetN, JoinRelationCPU<BitmapsetN>* > > unranked_rels;

        pthread_mutex_lock(args->mutex);
        while (*args->level <= level)
            pthread_cond_wait(args->start_thread_cond, args->mutex);

        level = *args->level;
        pthread_mutex_unlock(args->mutex);

        LOG_DEBUG("[%d] Starting iteration %d\n", args->id, level);

        uint_t<BitmapsetN> sid;
        uint_t<BitmapsetN> n_sets = BINOM(args->binoms, 
            args->info->n_rels, args->info->n_rels, level);

        // this should not overflow since the maximum value for n_sets is 
        // 32C16 (on 32 bits, ~600M), skip is small and the number of threads 
        // is small (worst case skip ~1000, n_threads ~100)
        while ((sid = args->next_sid->fetch_add(*args->skip)) < n_sets){
            BitmapsetN s = dpsub_unrank_sid<BitmapsetN>(
                            sid, level, args->info->n_rels, args->binoms);

            for (int i = 0; sid+i < n_sets && i < *args->skip; i++){
                BitmapsetN set = s << 1;
                LOG_DEBUG("[%d] Trying %u\n", args->id, set.toUint());
                if (is_connected(set, args->info->edge_table)){
                    LOG_DEBUG("[%d] %u is connected\n", args->id, set.toUint());
                    JoinRelationCPU<BitmapsetN> *join_rel = args->alg->enumerate_subsets(set);

                    Assert(join_rel != NULL);
                    unranked_rels.push_back(make_pair(set, join_rel));
                }

                s = s.nextPermutation();
            }
        }

        pthread_mutex_lock(args->mutex);

        for (auto &rel_pair : unranked_rels){
            args->memo->insert(rel_pair);
        }

        LOG_DEBUG("[%d] No more sets, waiting\n", args->id);
        (*args->n_waiting)++;
        if (*args->n_waiting == gpuqo_dpe_n_threads){
            pthread_cond_signal(args->thread_waiting_cond);
        }

        pthread_mutex_unlock(args->mutex);
    }

    return NULL;
}

template<typename BitmapsetN, typename memo_t>
class ParallelDPsubJoinFunction : public CPUJoinFunction<BitmapsetN, memo_t> {
public:
    ParallelDPsubJoinFunction(GpuqoPlannerInfo<BitmapsetN>* _info, 
        memo_t* _memo, CPUAlgorithm<BitmapsetN, memo_t>* _alg) 
        : CPUJoinFunction<BitmapsetN,memo_t>(_info, _memo, _alg) {}

    virtual JoinRelationCPU<BitmapsetN> *operator()(int level, bool try_swap,
                JoinRelationCPU<BitmapsetN> &left_rel, 
                JoinRelationCPU<BitmapsetN> &right_rel)
    {
        auto info = CPUJoinFunction<BitmapsetN, memo_t>::info;

        if (CPUJoinFunction<BitmapsetN, memo_t>::alg->check_join(level, left_rel, right_rel)){
            JoinRelationCPU<BitmapsetN> *join_rel1, *join_rel2;
            
            join_rel1 = make_join_relation<BitmapsetN,JoinRelationCPU<BitmapsetN>>(left_rel, right_rel, info);
            CPUJoinFunction<BitmapsetN, memo_t>::alg->post_join(level, true, *join_rel1, left_rel,  right_rel);
            if (try_swap){
                join_rel2 = make_join_relation<BitmapsetN,JoinRelationCPU<BitmapsetN>>(right_rel, left_rel, info);
                CPUJoinFunction<BitmapsetN, memo_t>::alg->post_join(level, true, *join_rel1, right_rel,  left_rel);
                if (join_rel1->cost.total < join_rel2->cost.total){
                    delete join_rel2;
                    return join_rel1;
                } else 
                    delete join_rel1;
                    return join_rel2;
            } else {
                return join_rel1;
            }
        } else {
            return NULL;
        }
    }
};

template<typename BitmapsetN>
void parallel_enumerate(GpuqoPlannerInfo<BitmapsetN>* info, 
                        level_hashtable<BitmapsetN>* memo, 
                        alg_t<BitmapsetN>* alg)
{
    ThreadArgs<BitmapsetN>* thread_args = 
                        new ThreadArgs<BitmapsetN>[gpuqo_dpe_n_threads];
    pthread_t *threads = new pthread_t[gpuqo_dpe_n_threads];
    pthread_cond_t thread_waiting_cond;
    pthread_cond_t start_thread_cond;
    pthread_mutex_t mutex;
    int level = 0;
    int n_waiting = 0;
    uint_t<BitmapsetN> skip;
    atomic<uint_t<BitmapsetN>> next_sid;
    int binoms_size = (info->n_rels+1)*(info->n_rels+1);
    thrust::host_vector<uint_t<BitmapsetN> > binoms(binoms_size);
    precompute_binoms<uint_t<BitmapsetN> >(binoms, info->n_rels);

    pthread_cond_init(&thread_waiting_cond, NULL);
    pthread_cond_init(&start_thread_cond, NULL);
    pthread_mutex_init(&mutex, NULL);
    
    for (int i=0; i<gpuqo_dpe_n_threads; i++){
        thread_args[i].id = i;
        thread_args[i].alg = alg;
        thread_args[i].memo = memo;
        thread_args[i].info = info;
        thread_args[i].binoms = thrust::raw_pointer_cast(binoms.data());
        thread_args[i].level = &level;
        thread_args[i].skip = &skip;
        thread_args[i].n_waiting = &n_waiting;
        thread_args[i].next_sid = &next_sid;
        thread_args[i].thread_waiting_cond = &thread_waiting_cond;
        thread_args[i].start_thread_cond = &start_thread_cond;
        thread_args[i].mutex = &mutex;

        LOG_DEBUG("Spawning thread %d\n", i);
        
        int ret = pthread_create(&threads[i], NULL, 
                                thread_function<BitmapsetN>, 
                                (void*) &thread_args[i]);

        if (ret != 0){
            perror("pthread_create: ");
            return;
        }
    }

    pthread_mutex_lock(&mutex);
    for (level=2; level<=info->n_iters; level++){
        next_sid = 0;
        uint_t<BitmapsetN> n_sets = BINOM(binoms, 
                                    info->n_rels, info->n_rels, level);
        skip = min(ceil_div(n_sets, gpuqo_dpe_n_threads),   
                    (uint_t<BitmapsetN>) gpuqo_cpu_dpsub_parallel_chunk_size);

        LOG_PROFILE("Starting iteration %d (skip=%u)\n", 
                    level, (unsigned int)skip);
        pthread_cond_broadcast(&start_thread_cond);

        while(n_waiting < gpuqo_dpe_n_threads)
            pthread_cond_wait(&thread_waiting_cond, &mutex);

        n_waiting = 0;

        LOG_DEBUG("End iteration %d\n", level);
    }
    pthread_mutex_unlock(&mutex);
    
    for (int i=0; i<gpuqo_dpe_n_threads; i++){
        LOG_DEBUG("Joining thread %d\n", i);
        pthread_join(threads[i], NULL);
    }

    delete[] thread_args;
    delete[] threads;
}

template<typename BitmapsetN>
QueryTree<BitmapsetN>* gpuqo_cpu_dpsub_generic_parallel(GpuqoPlannerInfo<BitmapsetN>* info, 
                                alg_t<BitmapsetN> *algorithm)
{
    
    DECLARE_TIMING(gpuqo_cpu_dpsub_parallel);
    START_TIMING(gpuqo_cpu_dpsub_parallel);

    level_hashtable<BitmapsetN> memo;
    QueryTree<BitmapsetN>* out = NULL;

    for(int i=0; i<info->n_rels; i++){
        JoinRelationCPU<BitmapsetN> *jr = new JoinRelationCPU<BitmapsetN>;
        jr->id = info->base_rels[i].id; 
        jr->left_rel_id = 0; 
        jr->left_rel_ptr = NULL; 
        jr->right_rel_id = 0; 
        jr->right_rel_ptr = NULL; 
        jr->cost = cost_baserel(info->base_rels[i]); 
        jr->width = info->base_rels[i].width; 
        jr->rows = info->base_rels[i].rows; 
        jr->edges = info->edge_table[i];
        memo.insert(std::make_pair(info->base_rels[i].id, jr));
    }

    ParallelDPsubJoinFunction<BitmapsetN, level_hashtable<BitmapsetN>> join_func(info, &memo, algorithm);

    algorithm->init(info, &memo, &join_func);
    
    parallel_enumerate(info, &memo, algorithm);

#ifdef GPUQO_PRINT_N_JOINS
    printf("The algorithm did %u joins over %u checks (1:%.2f)\n", 
            algorithm->get_n_joins(), algorithm->get_n_checks(),
           (double) algorithm->get_n_checks() / algorithm->get_n_joins());
#endif 

    BitmapsetN final_joinrel_id = BitmapsetN(0);
    
    if (info->n_rels == info->n_iters){ // normal DP
        for (int i = 0; i < info->n_rels; i++)
            final_joinrel_id |= info->base_rels[i].id;
    } else { // IDP
        float min_cost = INFF;
        for (auto iter=memo.begin(info->n_iters); iter != memo.end(info->n_iters); ++iter){
            if (iter->second->cost.total < min_cost){
                min_cost = iter->second->cost.total;
                final_joinrel_id = iter->first;
            }
        }
    }
    
    auto final_joinrel_pair = memo.find(final_joinrel_id);
    if (final_joinrel_pair != memo.end())
        build_query_tree(final_joinrel_pair->second, &out);

    // delete all dynamically allocated memory
    for (int i = 1; i <= info->n_rels; i++){
        auto bucket = memo.get_bucket(i);
        for (auto iter=bucket->begin(); iter != bucket->end(); ++iter){
            delete iter->second;
        }
    }

    STOP_TIMING(gpuqo_cpu_dpsub_parallel);
    PRINT_TIMING(gpuqo_cpu_dpsub_parallel);

    return out;
}

template QueryTree<Bitmapset32>* gpuqo_cpu_dpsub_generic_parallel<Bitmapset32>
        (GpuqoPlannerInfo<Bitmapset32>*, alg_t<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_dpsub_generic_parallel<Bitmapset64>
        (GpuqoPlannerInfo<Bitmapset64>*, alg_t<Bitmapset64>*);

        