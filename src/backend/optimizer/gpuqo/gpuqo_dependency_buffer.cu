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
DependencyBuffer<BitmapsetN>::DependencyBuffer(int n_rels, int capacity) 
        : n_rels(n_rels), capacity(capacity)
{
    lookup.reserve(capacity);

    free_nodes = new DepBufNode<BitmapsetN>[capacity];
    next_free_node = free_nodes;

    queues = new DepBufNode<BitmapsetN>*[n_rels+1];
    for (int i = 0; i <= n_rels; i++)
        queues[i] = NULL;

    unified_queue = NULL;
}

template<typename BitmapsetN>
void DependencyBuffer<BitmapsetN>::push(
            JoinRelationDPE<BitmapsetN> *join_rel,  
            JoinRelationDPE<BitmapsetN> *left_rel, 
            JoinRelationDPE<BitmapsetN> *right_rel)
{
    // push is not thread-safe
    int index = join_rel->id.size();

    Assert(next_free_node < free_nodes + capacity);

    DepBufNode<BitmapsetN> *new_node = next_free_node++;

    new_node->join_rel = join_rel;
    new_node->left_rel = left_rel;
    new_node->right_rel = right_rel;
    new_node->next_rel = new_node;
    new_node->prev_rel = new_node;
    new_node->next_join = new_node;
    new_node->prev_join = new_node;

    auto search_res = lookup.find(join_rel->id);

    if (queues[index] == NULL){
        queues[index] = new_node;

        lookup.emplace(join_rel->id, new_node);
        join_rel->num_entry++;
    } else if(search_res == lookup.end()){
        // add back
        new_node->next_rel = queues[index];
        new_node->prev_rel = queues[index]->prev_rel;
        queues[index]->prev_rel->next_rel = new_node;
        queues[index]->prev_rel = new_node;

        if (left_rel->num_entry.load() == 0 
                && right_rel->num_entry.load() == 0){
            // move to front
            queues[index] = new_node;
        }

        lookup.emplace(join_rel->id, new_node);
        join_rel->num_entry++;
    } else {
        DepBufNode<BitmapsetN> *node = search_res->second;

        // add back
        new_node->next_join = node;
        new_node->prev_join = node->prev_join;
        node->prev_join->next_join = new_node;
        node->prev_join = new_node;

        if (left_rel->num_entry.load() == 0 
                && right_rel->num_entry.load() == 0){
            // move to front
            if (node != node->next_rel){
                node->prev_rel->next_rel = new_node;
                node->next_rel->prev_rel = new_node;
                new_node->prev_rel = node->prev_rel;
                new_node->next_rel = node->next_rel;
            }
            if (queues[index] == node)
                queues[index] = new_node;

            search_res->second = new_node;
        }
    }
}

template<typename BitmapsetN>
void DependencyBuffer<BitmapsetN>::unify_queues()
{
    DepBufNode<BitmapsetN> *node = NULL;

    for (int index = 0; index <= n_rels; index++){
        if (queues[index] != NULL){
            if (node != NULL){
                node->prev_rel->next_rel = queues[index];
                node = queues[index];
            } else {
                node = queues[index];
                unified_queue = node;
            }
        }
    }

    if (node != NULL)
        node->prev_rel->next_rel = NULL;
}

template<typename BitmapsetN>
DepBufNode<BitmapsetN> *DependencyBuffer<BitmapsetN>::pop(){
    DepBufNode<BitmapsetN> *node;
    do{
        node = unified_queue.load();
    } while(node != NULL 
        && !unified_queue.compare_exchange_strong(node, node->next_rel)
    );

    return node;
}

template<typename BitmapsetN>
bool DependencyBuffer<BitmapsetN>::empty(){
    return unified_queue.load() == NULL;
}

template<typename BitmapsetN>
bool DependencyBuffer<BitmapsetN>::full(){
    return next_free_node >= free_nodes + capacity;
}

template<typename BitmapsetN>
void DependencyBuffer<BitmapsetN>::clear(){
    next_free_node = free_nodes;
    for (int i = 0; i <= n_rels; i++)
        queues[i] = NULL;
    lookup.clear();
    unified_queue = NULL;
}

template<typename BitmapsetN>
DependencyBuffer<BitmapsetN>::~DependencyBuffer(){
    delete[] free_nodes;
    delete[] queues;
}

template class DependencyBuffer<Bitmapset32>;
template class DependencyBuffer<Bitmapset64>;
