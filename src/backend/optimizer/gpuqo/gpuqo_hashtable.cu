/*------------------------------------------------------------------------
 *
 * gpuqo_hashtable.cu
 *      implementations for GPU Hash Table
 * 
 * Derived from: https://github.com/nosferalatu/SimpleGPUHashTable
 *
 * src/backend/optimizer/gpuqo/gpuqo_hashtable.cu
 *
 *-------------------------------------------------------------------------
 */

#include <vector>

#include "gpuqo_hashtable.cuh"
#include "gpuqo_debug.cuh"

// DEVICE FUNCTIONS IMPLEMENTATION

template <typename K, typename V, typename Kint>
__device__
V* HashTable<K,V,Kint>::lookup(K key){
    Kint slot = hash(key);
    while (true){
        if (keys[slot] == key){
            LOG_DEBUG("%u: found %u (%u)\n", key, slot, hash(key));
            return &values[slot];
        } else if (keys[slot] == EMPTY){
            LOG_DEBUG("%u: not found %u (%u)\n", key, slot, hash(key));
            return NULL;
        }

        LOG_DEBUG("%u: inc %u (%u)\n", key, slot, hash(key));

        slot = (slot + 1) & (capacity-1);
    }
}

template <typename K, typename V, typename Kint>
__device__
void HashTable<K,V,Kint>::insert(K key, V value){
    Kint slot = hash(key);
    while (true){
        K prev = atomicCAS(&keys[slot], EMPTY, key);
        if (prev == EMPTY || prev == key){
            LOG_DEBUG("%u: found %u (%u)\n", key, slot, hash(key));
            values[slot] = value;
            return;
        }

        LOG_DEBUG("%u: inc %u (%u)\n", key, slot, hash(key));

        slot = (slot + 1) & (capacity-1);
    }
}

// KERNELS IMPLEMENTATION

template<typename K, typename V, typename Kint>
__global__ 
void HashTable_insert(HashTable<K,V,Kint> hashtable, K* in_keys, V* in_values, size_t n)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < n)
    {
        K key = in_keys[threadid];
        V value = in_values[threadid];
        hashtable.insert(key, value);
        LOG_DEBUG("%u: inserted %u\n", threadid, key);
    }
}

template<typename K, typename V, typename Kint>
__global__
void HashTable_lookup(HashTable<K,V,Kint> hashtable, K* in_keys, V* out_values, size_t n)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < n)
    {
        K key = in_keys[threadid];
        V* val_p = hashtable.lookup(key);
        if (val_p)
            out_values[threadid] = *val_p;

        LOG_DEBUG("%u: looked up %u\n", threadid, key);  
    }
}

// HOST FUNCTIONS IMPLEMENTATION

template <typename K, typename V, typename Kint>
__host__
HashTable<K,V,Kint>::HashTable(size_t _capacity){
    // capacity must be a multiple of 2
    capacity = 1;
    while (capacity < _capacity)
        capacity *= 2;

        
    cudaMalloc(&keys, sizeof(K) * capacity);    
    cudaMemset(keys, 0xff, sizeof(K) * capacity);
    LOG_DEBUG("cudaMalloc(%llx, %u)\n", keys, sizeof(K) * capacity);
    
    cudaMalloc(&values, sizeof(V) * capacity);    
    LOG_DEBUG("cudaMalloc(%llx, %u)\n", values, sizeof(V) * capacity);

    cudaDeviceSynchronize();
#ifdef GPUQO_DEBUG
    std::vector<K> local_keys(capacity);
    cudaMemcpy(&local_keys[0], keys, sizeof(K)*capacity, cudaMemcpyDeviceToHost);
    LOG_DEBUG("hashtable before insert:\n");
    DUMP_VECTOR(local_keys.begin(), local_keys.end());
#endif
}

template <typename K, typename V, typename Kint>
__host__
void HashTable<K,V,Kint>::lookup(K* in_keys, V* out_values, size_t n){

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, HashTable_lookup<K,V,Kint>, 0, 0);

    // Lookup all the keys on the hash table
    int gridsize = (n + threadblocksize - 1) / threadblocksize;
    HashTable_lookup<K,V,Kint><<<gridsize, threadblocksize>>>(*this, in_keys, out_values, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err){
        printf("CUDA ERROR! %s: %s\n", 
            cudaGetErrorName(err),
            cudaGetErrorString(err)
        );
    }
}

template <typename K, typename V, typename Kint>
__host__
void HashTable<K,V,Kint>::insert(K* in_keys, V* in_values, size_t n){
    LOG_DEBUG("HashTable::insert(%llx, %llx, %u)\n", in_keys, in_values, n);

#ifdef GPUQO_DEBUG
    std::vector<K> local_keys(capacity);
    cudaMemcpy(&local_keys[0], keys, sizeof(K)*capacity, cudaMemcpyDeviceToHost);
    LOG_DEBUG("hashtable before insert:\n");
    DUMP_VECTOR(local_keys.begin(), local_keys.end());
#endif
    
    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, HashTable_insert<K,V,Kint>, 0, 0);

    // Insert all the keys into the hash table
    int gridsize = (n + threadblocksize - 1) / threadblocksize;
    HashTable_insert<K,V,Kint><<<gridsize, threadblocksize>>>(*this, in_keys, in_values, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err){
        printf("CUDA ERROR! %s: %s\n", 
            cudaGetErrorName(err),
            cudaGetErrorString(err)
        );
    }
}


template <typename K, typename V, typename Kint>
__host__
V HashTable<K,V,Kint>::get(K key){
    V val;

    K* dev_key;
    cudaMalloc(&dev_key, sizeof(K));
    cudaMemcpy(dev_key, &key, sizeof(K), cudaMemcpyHostToDevice);

    V* dev_val;
    cudaMalloc(&dev_val, sizeof(V));

    lookup(dev_key, dev_val, 1);

    cudaMemcpy(&val, dev_val, sizeof(V), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err){
        printf("CUDA ERROR! %s: %s\n", 
            cudaGetErrorName(err),
            cudaGetErrorString(err)
        );
    }

    return val;
}

template <typename K, typename V, typename Kint>
__host__
void HashTable<K,V,Kint>::free(){
    cudaFree(keys);    
    cudaFree(values);    
}

template<>
__device__
unsigned int HashTable<uint32_t, JoinRelation, unsigned int>::hash(uint32_t k){
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity-1);
}




// explicit specification
template class HashTable<RelationID,JoinRelation,unsigned int>;
