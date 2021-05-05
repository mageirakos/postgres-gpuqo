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
#include "gpuqo_debug.cuh"

template <typename K, typename V>
using HashTableKV = thrust::tuple<K,V>;

template <typename K, typename V>
using HashTableIterator = thrust::zip_iterator<thrust::tuple<thrust::device_ptr<K>, thrust::device_ptr<V> > >;

template <typename K, typename V>
class HashTable{
private:
    K* keys;
    V* values;
    size_t capacity;
    size_t max_capacity;

    // Upper bound on the number of stored elements
    size_t n_elems_ub;

    __device__
    size_t hash(K key){
        return key.hash() & (capacity-1);
    }

    __host__
    void debugDump();

    __host__
    bool deviceMalloc();

    __host__
    bool deviceErrorCheck();

    __host__ 
    bool _insert(K* keys, V* values, size_t n);

public:
    __host__
    HashTable(size_t initial_size, size_t max_capacity);

    __device__ 
    void insert(K key, V value);
    __host__ 
    bool insert(K* keys, V* values, size_t n);


    __device__ 
    V* lookup(K key);

    __host__ 
    bool lookup(K* keys, V* values, size_t n);

    __host__
    V get(K key);

    __host__ __device__ __forceinline__
    size_t getCapacity(){ return capacity; }

    __host__
    bool resize(size_t capacity);

    __host__
    void free();

    __host__
    HashTableIterator<K,V> begin(){
        return thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::device_pointer_cast(keys),
                thrust::device_pointer_cast(values)
            )
        );
    }

    __host__
    HashTableIterator<K,V> end(){
        return thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::device_pointer_cast(keys+capacity),
                thrust::device_pointer_cast(values+capacity)
            )
        );
    }
};

template<typename K, typename V> 
__host__
HashTable<K,V>* createHashTable(size_t capacity);

template<typename BitmapsetN>
using HashTableDpsub = HashTable<BitmapsetN,JoinRelation<BitmapsetN> >;

template<typename BitmapsetN>
using HashTableKVDpsub = HashTableKV<BitmapsetN,JoinRelation<BitmapsetN> >;

template<typename BitmapsetN>
using HashTableIteratorDpsub = HashTableIterator<BitmapsetN,JoinRelation<BitmapsetN> >;

// DEVICE FUNCTIONS IMPLEMENTATION

template <typename K, typename V>
__device__
V* HashTable<K,V>::lookup(K key){
    size_t slot = hash(key);
    size_t first_slot = slot;
    do {
        if (keys[slot] == key){
            LOG_DEBUG("%u: found %lu (%lu)\n", key.toUint(), slot, first_slot);
            return &values[slot];
        } else if (keys[slot].empty()){
            // NB: elements cannot be deleted!
            printf("%u: not found %lu (%lu)\n", key.toUint(), slot, first_slot);
            return NULL;
        }

        LOG_DEBUG("%u: inc %lu (%lu)\n", key.toUint(), slot, first_slot);

        slot = (slot + 1) & (capacity-1);
    } while (slot != first_slot);

    printf("%u: not found %lu (%lu), table is full\n", key.toUint(), slot, first_slot);

    // I checked all available positions
    return NULL;
}

template <typename K, typename V>
__device__
void HashTable<K,V>::insert(K key, V value){
    size_t slot = hash(key);
    size_t first_slot = slot;
    do {
        K prev = atomicCAS(&keys[slot], K(0), key);
        if (prev.empty() || prev == key){
            LOG_DEBUG("%u: found %u (%u)\n", key.toUint(), slot, hash(key));
            values[slot] = value;
            return;
        }

        LOG_DEBUG("%u: inc %u (%u)\n", key.toUint(), slot, hash(key));

        slot = (slot + 1) & (capacity-1);
    } while (slot != first_slot);

    // I checked all available positions
    // table is full
    assert(false);
}

#endif