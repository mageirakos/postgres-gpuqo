/*-------------------------------------------------------------------------
 *
 * gpuqo_dependency_buffer.cu
 *	  definition of DependencyBuffer's methods 
 *
 * src/include/optimizer/gpuqo_dependency_buffer.cu
 *
 *-------------------------------------------------------------------------
 */

#include "optimizer/gpuqo_dependency_buffer.cuh"

DependencyBuffer::DependencyBuffer(int n_rels) 
        : n_rels(n_rels) {
    pthread_mutex_init(&mutex, NULL);
    queues = new depbuf_queue_t[n_rels*n_rels];
    first_non_empty = n_rels*n_rels;
}

void DependencyBuffer::push(JoinRelationDPE *join_rel,  
                        JoinRelationDPE *left_rel, JoinRelationDPE *right_rel){
    // push is not thread-safe
    int left_size = BMS64_SIZE(join_rel->left_relation_id);
    int right_size = BMS64_SIZE(join_rel->right_relation_id);
    int big_size = std::max(left_size, right_size);
    int small_size = std::min(left_size, right_size);
    int index = big_size * n_rels + small_size;

    depbuf_queue_t::iterator entry = queues[index].end();
    for (depbuf_queue_t::iterator iter = queues[index].begin(); iter != queues[index].end(); ++iter)
    {
        if (iter->first == join_rel){ // I can check address here
            entry = iter;
            break;
        }
    }

    if (entry == queues[index].end()){
        int num = join_rel->num_entry.fetch_add(1, std::memory_order_consume);
        Assert(num >= 0);
        
        depbuf_entry_t temp = std::make_pair(
            join_rel, 
            depbuf_list_t()
        );

        if (left_rel->num_entry.load(std::memory_order_consume) == 0 
                && right_rel->num_entry.load(std::memory_order_consume) == 0){
            queues[index].push_front(temp);
            entry = queues[index].begin();
        } else {
            queues[index].push_back(temp);
            entry = --queues[index].end();
        }
    
        if (index < first_non_empty)
            first_non_empty = index;
    }

    entry->second.push_back(std::make_pair(left_rel, right_rel));
}

depbuf_entry_t DependencyBuffer::pop(){
    depbuf_entry_t out;
    int num;
    out.first = NULL; 

    pthread_mutex_lock(&mutex);
    
    if (empty())
        goto exit;

    out = queues[first_non_empty].front();
    queues[first_non_empty].pop_front();

    num = out.first->num_entry.fetch_sub(1, std::memory_order_release);
    Assert(num > 0);

    while (queues[first_non_empty].empty() && first_non_empty < n_rels*n_rels)
        first_non_empty++;
    // stop if found non-empty queue or first_non_empty = n_rels*n_rels

exit:
    pthread_mutex_unlock(&mutex);
    return out;
}

bool DependencyBuffer::empty(){
    return first_non_empty >= n_rels*n_rels;
}

DependencyBuffer::~DependencyBuffer(){
    delete queues;
    pthread_mutex_destroy(&mutex);
}
