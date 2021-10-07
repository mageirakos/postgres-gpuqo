/*------------------------------------------------------------------------
 *
 * gpuqo_cpu_goo.cu
 *      Greedy Operator Ordering approximate algorithm
 *
 * src/backend/optimizer/gpuqo/gpuqo_dpccp.cu
 *
 *-------------------------------------------------------------------------
 */

#include <queue>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_cost.cuh"
#include "gpuqo_filter.cuh"

using namespace std;

template<typename BitmapsetN>
struct CompareQueryTree {
    bool operator()(QueryTree<BitmapsetN>* l, QueryTree<BitmapsetN>* r) 
    {
        if (l != NULL && r != NULL)
            return l->rows > r->rows;
        else 
            return r != NULL;
    }
};

template<typename BitmapsetN>
static QueryTree<BitmapsetN>* 
join(QueryTree<BitmapsetN> &left, QueryTree<BitmapsetN> &right,
    GpuqoPlannerInfo<BitmapsetN>* info) 
{
    QueryTree<BitmapsetN>* qt = new QueryTree<BitmapsetN>;
    qt->id = left.id | right.id;
    qt->rows = estimate_join_rows(left, right, info);
    qt->cost.startup = 0;
    qt->cost.total = qt->rows + left.cost.total + right.cost.total;
    qt->left = &left;
    qt->right = &right;
    qt->width = left.width + right.width;
    return qt;
}

/* gpuqo_cpu_goo
 *
 *	 Gready Operator Ordering approximation
 */
template<typename BitmapsetN>
QueryTree<BitmapsetN>*
gpuqo_cpu_goo(GpuqoPlannerInfo<BitmapsetN>* info)
{
    vector<QueryTree<BitmapsetN>* > heap;
    map<BitmapsetN, QueryTree<BitmapsetN>* > relations;
    CompareQueryTree<BitmapsetN> compare;

    LOG_DEBUG("EDGES:\n");
    for (int i = 0; i < info->n_rels; i++) {
        QueryTree<BitmapsetN>* qt = new QueryTree<BitmapsetN>;
        qt->id = info->base_rels[i].id;
        qt->rows = info->base_rels[i].rows;
        qt->cost.startup = 0;
        qt->cost.total = info->base_rels[i].rows;
        qt->left = NULL;
        qt->right = NULL;
        qt->width = info->base_rels[i].width;
        relations.insert(make_pair(qt->id, qt));
        LOG_DEBUG("%d: %u\n", i+1, info->edge_table[i].toUint());
    }

    for (auto i = relations.begin(); i != relations.end(); i++) {
            for (auto j = i; j != relations.end(); j++) {
                LOG_DEBUG("Check %u %u\n", 
                            i->first.toUint(), j->first.toUint());
                if (are_valid_pair(i->first, j->first, info)) {
                    QueryTree<BitmapsetN>* qt = join(*i->second, *j->second, info);
                    heap.push_back(qt);
                    push_heap(heap.begin(), heap.end(), compare);
                    LOG_DEBUG("Pushed %u (%.0f) in heap\n", 
                                qt->id.toUint(), qt->rows);
                }
        }
    }
    
    QueryTree<BitmapsetN> *smallest = NULL;
    do {
        // pop join with smallest result set
        Assert(!heap.empty());
        pop_heap(heap.begin(), heap.end(), compare);
        smallest = heap.back();
        heap.pop_back();
        LOG_DEBUG("Popped %u (%.0f) from heap: ", 
                    smallest->id.toUint(), smallest->rows);

        // check that smallest is still valid
        // ie, that its left and right children are still in the relations set
        auto left_pos = relations.find(smallest->left->id);
        auto right_pos = relations.find(smallest->right->id);
        if (left_pos == relations.end() || right_pos == relations.end()) {
            LOG_DEBUG("discarded\n"); 
            delete smallest;
            smallest = NULL;
            continue;
        }

        LOG_DEBUG("valid\n"); 
        
        // remove left and right relations
        relations.erase(left_pos);
        relations.erase(right_pos);

        // add all its combinations with other relations to the heap
        for (auto j = relations.begin(); j != relations.end(); j++) {
            if (are_valid_pair(smallest->id, j->first, info)) {
                QueryTree<BitmapsetN>* qt = join(*smallest, *j->second, info);
                heap.push_back(qt);
                push_heap(heap.begin(), heap.end(), compare);
                LOG_DEBUG("Pushed %u (%.0f) in heap\n", 
                                qt->id.toUint(), qt->rows);
            }
        }

        // insert smallest in relations
        relations.insert(make_pair(smallest->id, smallest));
    } while(smallest == NULL || smallest->id.size() < info->n_rels);

    // free all remaining pairs in the heap
    for (QueryTree<BitmapsetN>* qt:heap)
        delete qt;

    // free all remaining pairs in relations, except smallest, which will be
    // returned
    for (auto i = relations.begin(); i != relations.end(); i++)
        if (i->second != smallest)
            delete i->second;

    return smallest;
}

template QueryTree<Bitmapset32>* gpuqo_cpu_goo<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32>*);
template QueryTree<Bitmapset64>* gpuqo_cpu_goo<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64>*);
template QueryTree<BitmapsetDynamic>* gpuqo_cpu_goo<BitmapsetDynamic>(GpuqoPlannerInfo<BitmapsetDynamic>*);
