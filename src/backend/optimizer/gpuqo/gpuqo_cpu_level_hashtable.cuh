/*-------------------------------------------------------------------------
 *
 * gpuqo_cpu_level_hashtable.cuh
 *	  class providing unordered_map interface but using multiple maps 
 *    internally, one per level, in order to remove dependencies between
 *    joinsets of different size
 *
 * src/include/optimizer/gpuqo_cpu_level_hashtable.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_CPU_LEVEL_HASHTABLE_CUH
#define GPUQO_CPU_LEVEL_HASHTABLE_CUH

#include <unordered_map>
#include <utility>

#include "gpuqo_cpu_common.cuh"

using namespace std;

template<typename BitmapsetN>
using iterator_t = typename hashtable_memo_t<BitmapsetN>::iterator;

template<typename BitmapsetN>
class level_hashtable{
private:
    unordered_map<BitmapsetN, JoinRelationCPU<BitmapsetN>*> buckets[BitmapsetN::SIZE+1];

public:
    iterator_t<BitmapsetN> find(const BitmapsetN& k){
        auto res = buckets[k.size()].find(k);
        if (res == buckets[k.size()].end())
            return end();
        else 
            return res;
    }

    pair<iterator_t<BitmapsetN>,bool> insert(const pair<BitmapsetN, JoinRelationCPU<BitmapsetN>*>& v){
        return buckets[v.first.size()].insert(v);
    }

    iterator_t<BitmapsetN> begin(int i){
        return buckets[i].begin();
    }

    iterator_t<BitmapsetN> end(int i){
        return buckets[i].end();
    }

    iterator_t<BitmapsetN> end(){
        return buckets[BitmapsetN::SIZE].end();
    }

    unordered_map<BitmapsetN, JoinRelationCPU<BitmapsetN>*> *get_bucket(int size){
        return &buckets[size];
    }
};

#endif