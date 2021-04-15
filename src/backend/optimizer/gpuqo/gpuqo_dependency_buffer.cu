/*-------------------------------------------------------------------------
 *
 * gpuqo_dependency_buffer.cu
 *	  definition of DependencyBuffer's methods 
 *
 * src/include/optimizer/gpuqo_dependency_buffer.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_dependency_buffer.cuh"

template<typename BitmapsetN>
DependencyBuffer<BitmapsetN>::DependencyBuffer(int n_rels) 
        : n_rels(n_rels) 
{
    pthread_mutex_init(&mutex, NULL);
    queue_lookup_pairs = new std::pair<depbuf_queue_t<BitmapsetN>, depbuf_lookup_t<BitmapsetN> >[n_rels*n_rels];
    first_non_empty = n_rels*n_rels;
}

template<typename BitmapsetN>
void DependencyBuffer<BitmapsetN>::push(
            JoinRelationDPE<BitmapsetN> *join_rel,  
            JoinRelationDPE<BitmapsetN> *left_rel, 
            JoinRelationDPE<BitmapsetN> *right_rel)
{
    // push is not thread-safe
    int left_size = left_rel->id.size();
    int right_size = right_rel->id.size();
    int big_size = std::max(left_size, right_size);
    int small_size = std::min(left_size, right_size);
    int index = big_size * n_rels + small_size;

    auto id_entry_pair = queue_lookup_pairs[index].second.find(join_rel->id);
    depbuf_entry_t<BitmapsetN>* entry;
    if (id_entry_pair == queue_lookup_pairs[index].second.end()){
        int num = join_rel->num_entry.fetch_add(1, std::memory_order_consume);
        Assert(num >= 0);
        
        depbuf_entry_t<BitmapsetN> temp = std::make_pair(
            join_rel, 
            new join_list_t<BitmapsetN>
        );

        if (left_rel->num_entry.load(std::memory_order_consume) == 0 
                && right_rel->num_entry.load(std::memory_order_consume) == 0){
            queue_lookup_pairs[index].first.push_front(temp);
            entry = &(*queue_lookup_pairs[index].first.begin());
        } else {
            queue_lookup_pairs[index].first.push_back(temp);
            entry = &(*queue_lookup_pairs[index].first.rbegin());
        }

        queue_lookup_pairs[index].second.insert(std::make_pair(
            join_rel->id, entry
        ));
    
        if (index < first_non_empty)
            first_non_empty = index;
    } else {
        entry = id_entry_pair->second;
    }

    entry->second->push_back(std::make_pair(left_rel, right_rel));
}

template<typename BitmapsetN>
depbuf_entry_t<BitmapsetN> DependencyBuffer<BitmapsetN>::pop(){
    depbuf_entry_t<BitmapsetN> out;
    int num __attribute__((unused));
    out.first = NULL; 

    pthread_mutex_lock(&mutex);
    
    if (empty())
        goto exit;

    out = queue_lookup_pairs[first_non_empty].first.front();
    queue_lookup_pairs[first_non_empty].first.pop_front();

    num = out.first->num_entry.fetch_sub(1, std::memory_order_release);
    Assert(num > 0);

    while (queue_lookup_pairs[first_non_empty].first.empty() 
            && first_non_empty < n_rels*n_rels)
        first_non_empty++;
    // stop if found non-empty queue or first_non_empty = n_rels*n_rels

exit:
    pthread_mutex_unlock(&mutex);
    return out;
}

template<typename BitmapsetN>
bool DependencyBuffer<BitmapsetN>::empty(){
    return first_non_empty >= n_rels*n_rels;
}

template<typename BitmapsetN>
void DependencyBuffer<BitmapsetN>::clear(){
    for (int i = 0; i < n_rels*n_rels; i++){
        queue_lookup_pairs[i].first.clear();
        queue_lookup_pairs[i].second.clear();
    }
    first_non_empty = n_rels*n_rels;
}

template<typename BitmapsetN>
size_t DependencyBuffer<BitmapsetN>::size(){
    size_t s = 0;
    for (int i = 0; i < n_rels*n_rels; i++){
        s += queue_lookup_pairs[i].first.size();
    }
    return s;
}

template<typename BitmapsetN>
DependencyBuffer<BitmapsetN>::~DependencyBuffer(){
    delete queue_lookup_pairs;
    pthread_mutex_destroy(&mutex);
}

template class DependencyBuffer<Bitmapset32>;
template class DependencyBuffer<Bitmapset64>;