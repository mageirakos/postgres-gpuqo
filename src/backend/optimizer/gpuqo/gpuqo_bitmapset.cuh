/*-------------------------------------------------------------------------
 *
 * gpuqo_bitmapset.cuh
 *	  declaration of function manipulating bitmapsets.
 *
 * src/optimizer/gpuqo/gpuqo_bitmapset.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_BITMAPSET_CUH
#define GPUQO_BITMAPSET_CUH

#include <functional>
#include <type_traits> 

#include "gpuqo_bit_manipulation.cuh"
#include "gpuqo_debug.cuh"

#define N_TO_SHOW 64

template<typename T>
class GpuqoBitmapset{
public: 
    T bits;

    typedef T type;
    
    typedef typename std::conditional<sizeof(T) <= 4,
                            uint2,
                            ulong2>::type type2;

    static constexpr int SIZE = sizeof(T)*8;

    __host__ __device__
    inline static GpuqoBitmapset<T> nth(unsigned n){
        return ((T)1) << n;
    }
    
    __host__ __device__
    inline GpuqoBitmapset<T>(){}
    
    __host__ __device__
    inline GpuqoBitmapset<T>(const T &_bits){
        bits = _bits;
    }
    
    __host__ __device__
    inline GpuqoBitmapset<T>(const GpuqoBitmapset<T> &other){
        bits = other.bits;
    }
    
    __host__ __device__
    inline GpuqoBitmapset<T>(const GpuqoBitmapset<T> volatile &other){
        bits = other.bits;
    }
    
    __host__ __device__
    inline unsigned size() const{
        return popc(bits);
    }

    __host__ __device__
    inline bool empty() const{
        return bits == 0;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> cmp() const{
        return ~bits;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> lowest() const{
        return blsi(bits);
    }

    __host__ __device__
    inline unsigned lowestPos() const{
        return ffs(bits)-1;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> highest() const{
        return nth(highestPos());
    }
    
    __host__ __device__
    inline unsigned highestPos() const{
        return sizeof(T)*8 - clz(bits) - 1;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> allLower() const{
        Assert(!empty());
        return lowest().bits-1;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> allLowerInc() const{
        Assert(!empty());
        return bits | (bits-1);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> unionSet(const GpuqoBitmapset<T> &other) const{
        return bits | other.bits;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> intersectionSet(const GpuqoBitmapset<T> &other) const{
        return bits & other.bits;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> differenceSet(const GpuqoBitmapset<T> &other) const{
        return intersectionSet(other.cmp());
    }

    __host__ __device__
    inline bool intersects(const GpuqoBitmapset<T> &other) const{
        return !intersectionSet(other).empty();
    }

    __host__ __device__
    inline bool isSubset(const GpuqoBitmapset<T> &other) const{
        return intersectionSet(other) == *this;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> set(unsigned n){
        return *this = unionSet(nth(n));
    }

    __host__ __device__
    inline GpuqoBitmapset<T> unset(unsigned n){
        return *this = differenceSet(nth(n));
    }

    __host__ __device__
    inline bool isSet(unsigned n) const{
        return intersects(nth(n));
    }

    __host__ __device__
    inline GpuqoBitmapset<T> nextPermutation() const{
        T t = (bits | (bits - 1)) + 1;  
        return t | (((blsi(t) / blsi(bits)) >> 1) - 1);  
    }

    __host__ __device__
    inline size_t hash() const;

    __host__ __device__
    inline unsigned toUint() const{
        return (unsigned) bits;
    }

    __host__ __device__
    inline unsigned long long toUlonglong() const{
        return (unsigned long long) bits;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator|(const GpuqoBitmapset<T> &other) const{
        return unionSet(other);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator&(const GpuqoBitmapset<T> &other) const{
        return intersectionSet(other);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator^(const GpuqoBitmapset<T> &other) const{
        return bits ^ other.bits;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator-(const GpuqoBitmapset<T> &other) const{
        return differenceSet(other);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator<<(const unsigned x) const{
        return bits << x;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator>>(const unsigned x) const{
        return bits >> x;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator++(int) {
        return bits++;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator~() const{
        return cmp();
    }

    __host__ __device__
    inline GpuqoBitmapset<T> &operator|=(const GpuqoBitmapset<T> &other){
        return *this = unionSet(other);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> &operator&=(const GpuqoBitmapset<T> &other){
        return *this = intersectionSet(other);
    }

    __host__ __device__
    inline GpuqoBitmapset<T> &operator^=(const GpuqoBitmapset<T> &other){
        bits ^= other.bits;
        return *this;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> &operator-=(const GpuqoBitmapset<T> &other){
        return *this = differenceSet(other);
    }

    __host__ __device__
    inline bool operator==(const GpuqoBitmapset<T> &other) const{
        return bits == other.bits;
    }

    __host__ __device__
    inline bool operator!=(const GpuqoBitmapset<T> &other) const{
        return bits != other.bits;
    }

    __host__ __device__
    inline bool operator<(const GpuqoBitmapset<T> &other) const{
        return bits < other.bits;
    }

    __host__ __device__
    inline bool operator<=(const GpuqoBitmapset<T> &other) const{
        return bits < other.bits;
    }

    __host__ __device__
    inline bool operator>(const GpuqoBitmapset<T> &other) const{
        return bits > other.bits;
    }

    __host__ __device__
    inline bool operator>=(const GpuqoBitmapset<T> &other) const{
        return bits > other.bits;
    }

    #pragma hd_warning_disable // WTF
    __host__ __device__
    inline GpuqoBitmapset<T> &operator=(const GpuqoBitmapset<T> &other){
        bits = other.bits;
        return *this;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> &operator=(const GpuqoBitmapset<T> volatile &other){
        bits = other.bits;
        return *this;
    }

    __host__ __device__
    inline GpuqoBitmapset<T> operator=(const GpuqoBitmapset<T> other) volatile{
        bits = other.bits;
        return *this;
    }
};


template<>
__host__ __device__
inline size_t GpuqoBitmapset<unsigned int>::hash() const{
    unsigned int x = bits;
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;        
}

template<>
__host__ __device__
inline size_t GpuqoBitmapset<unsigned long long int>::hash() const{
    unsigned long long int x = bits;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdLLU;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53LLU;
    x ^= x >> 33;     
    return x;
}

template<typename T>
__host__ __device__
inline GpuqoBitmapset<T> nextSubset(const GpuqoBitmapset<T> &subset, const GpuqoBitmapset<T> &set){
    return GpuqoBitmapset<T>(set.bits & (subset.bits - set.bits));
}

template<typename T>
__host__ __device__
inline GpuqoBitmapset<T> expandToMask(const GpuqoBitmapset<T> &val, const GpuqoBitmapset<T> &mask){
    return GpuqoBitmapset<T>(pdep(val.bits, mask.bits));
}

template <typename Type>
__device__
inline GpuqoBitmapset<Type> atomicCAS(GpuqoBitmapset<Type> *address, GpuqoBitmapset<Type> compare, GpuqoBitmapset<Type> val){
    return atomicCAS(&address->bits, compare.bits, val.bits);
}

template <typename Type>
__device__
inline GpuqoBitmapset<Type> atomicOr(GpuqoBitmapset<Type> *address, GpuqoBitmapset<Type> val){
    return atomicOr(&address->bits, val.bits);
}

namespace std {
    template<typename T>
    struct hash<GpuqoBitmapset<T> > {
        inline size_t operator()(const GpuqoBitmapset<T>& x) const {
            return hash<T>{}(x.bits);
        }
    };
}

typedef GpuqoBitmapset<unsigned int> Bitmapset32;
typedef GpuqoBitmapset<unsigned long long int> Bitmapset64;

template<typename BitmapsetN>
using uint_t = typename BitmapsetN::type;

template<typename BitmapsetN>
using uint2_t = typename BitmapsetN::type2;

#endif							/* GPUQO_BITMAPSET_CUH */
