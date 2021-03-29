/*------------------------------------------------------------------------
 *
 * gpuqo_hashtable.cuh
 *      definitions for GPU Hash Table
 * 
 * Derived from: https://github.com/nosferalatu/SimpleGPUHashTable
 *
 * src/backend/optimizer/gpuqo/gpuqo_hashtable.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_HASHTABLE_CUH
#define GPUQO_HASHTABLE_CUH

#include <limits>

#include "gpuqo.cuh"

template <typename K, typename V, typename Kint>
class HashTable{
private:
    K* keys;
    V* values;
    size_t capacity;
    size_t max_capacity;

    // Upper bound on the number of stored elements
    size_t n_elems_ub;

    __device__
    Kint hash(K key);

    __host__
    void debugDump();

    __host__
    void deviceMalloc();

    __host__
    void deviceErrorCheck();

    __host__ 
    void _insert(K* keys, V* values, size_t n);

public:
    static const K EMPTY = std::numeric_limits<K>::max();

    __host__
    HashTable(size_t initial_size, size_t max_capacity);

    __device__ 
    void insert(K key, V value);
    __host__ 
    void insert(K* keys, V* values, size_t n);


    __device__ 
    V* lookup(K key);

    __host__ 
    void lookup(K* keys, V* values, size_t n);

    __host__
    V get(K key);

    __host__ __device__ __forceinline__
    size_t getCapacity(){ return capacity; }

    __host__
    void resize(size_t capacity);

    __host__
    void free();
};

template<typename K, typename V, typename Kint> 
__host__
HashTable<K,V,Kint>* createHashTable(size_t capacity);

typedef HashTable<RelationID,JoinRelation,unsigned int> HashTable32bit;

#endif