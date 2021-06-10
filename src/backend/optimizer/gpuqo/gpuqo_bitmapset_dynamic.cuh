/*-------------------------------------------------------------------------
 *
 * gpuqo_bitmapset_dynamic.cuh
 *	  declaration of function manipulating variable length bitmapsets.
 *
 * src/optimizer/gpuqo/gpuqo_bitmapset_dynamic.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_BITMAPSET_DYNAMIC_CUH
#define GPUQO_BITMAPSET_DYNAMIC_CUH

#include <functional>
#include <type_traits> 

#include "gpuqo_bit_manipulation.cuh"
#include "gpuqo_bitmapset.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_postgres.cuh"

#define BITMAPWORD_FULL ~((bitmapword)0)

class BitmapsetDynamic{
public: 
    Bitmapset *bms;

    static constexpr int SIZE = 1024;

    static BitmapsetDynamic nth(unsigned n){
        return BitmapsetDynamic(bms_make_singleton(n));
    }
    
    BitmapsetDynamic(){
        bms = NULL;
    }
    
    ~BitmapsetDynamic(){
        bms_free(bms);
    }
    
    BitmapsetDynamic(const BitmapsetDynamic &other){
        bms = bms_copy(other.bms);
    }
    
    BitmapsetDynamic(Bitmapset* _bms){
        bms = _bms;
    }
    
    unsigned size() const{
        return bms_num_members(bms);
    }
    
    unsigned nbits() const{
        if (bms == NULL)
            return 0;

        return bms->nwords*BITS_PER_BITMAPWORD;
    }

    bool empty() const{
        return bms_is_empty(bms);
    }

    BitmapsetDynamic lowest() const{
        Bitmapset *out = bms_copy(bms);
        int i;
        for (i = 0; i < bms->nwords; i++){
            if (bms->words[i] != 0){
                out->words[i] = blsi(bms->words[i]);
                break;
            } else {
                out->words[i] = 0;
            }
        }
        i++;
        for (; i < bms->nwords; i++){
            out->words[i] = 0;
        }
        return BitmapsetDynamic(out);
    }

    unsigned lowestPos() const{
        for (int i = 0; i < bms->nwords; i++){
            if (bms->words[i] != 0){
                return ffs(bms->words[i])-1 + i*64;
            }
        }
        return size();
    }

    BitmapsetDynamic allLower() const{
        Bitmapset *out = bms_copy(bms);

        if (bms == NULL)
            return NULL;

        int i;
        for (i = 0; i < bms->nwords; i++){
            if (bms->words[i] != 0){
                out->words[i] = blsi(bms->words[i])-1;
                break;
            } else {
                out->words[i] = BITMAPWORD_FULL;
            }
        }
        i++;
        for (; i < bms->nwords; i++){
            out->words[i] = 0;
        }
        return BitmapsetDynamic(out);
    }

    BitmapsetDynamic unionSet(const BitmapsetDynamic &other) const{
        return BitmapsetDynamic(bms_union(bms, other.bms));
    }

    BitmapsetDynamic intersectionSet(const BitmapsetDynamic &other) const{
        return BitmapsetDynamic(bms_intersect(bms, other.bms));
    }

    BitmapsetDynamic differenceSet(const BitmapsetDynamic &other) const{
        return BitmapsetDynamic(bms_difference(bms, other.bms));
    }

    bool intersects(const BitmapsetDynamic &other) const{
        return bms_overlap(bms, other.bms);
    }

    bool isSubset(const BitmapsetDynamic &other) const{
        return bms_is_subset(bms, other.bms);
    }

    BitmapsetDynamic set(unsigned n){
        bms = bms_add_member(bms, n);
        return *this;
    }

    BitmapsetDynamic unset(unsigned n){
        bms = bms_del_member(bms, n);
        return *this;
    }

    bool isSet(unsigned n) const{
        return bms_is_member(n, bms);
    }

    size_t hash() const {
        return bms_hash_value(bms);
    }

    unsigned toUint() const{
        if (bms != NULL)
            return (unsigned) bms->words[0];
        else 
            return 0;
    }

    BitmapsetDynamic operator|(const BitmapsetDynamic &other) const{
        return unionSet(other);
    }

    BitmapsetDynamic operator&(const BitmapsetDynamic &other) const{
        return intersectionSet(other);
    }

    BitmapsetDynamic operator-(const BitmapsetDynamic &other) const{
        return differenceSet(other);
    }

    BitmapsetDynamic &operator|=(const BitmapsetDynamic &other){
        return *this = unionSet(other);
    }

    BitmapsetDynamic &operator&=(const BitmapsetDynamic &other){
        return *this = intersectionSet(other);
    }

    BitmapsetDynamic &operator-=(const BitmapsetDynamic &other){
        return *this = differenceSet(other);
    }

    bool operator==(const BitmapsetDynamic &other) const{
        return bms_equal(bms, other.bms);
    }

    bool operator!=(const BitmapsetDynamic &other) const{
        return !(*this == other);
    }

    BitmapsetDynamic& operator=(const BitmapsetDynamic& other) {
        bms_free(bms);
        bms = bms_copy(other.bms);
        return *this;
    }
};


static 
BitmapsetDynamic expandToMask(const BitmapsetDynamic &val, const BitmapsetDynamic &mask){
    int m = 0, v = 0;
    BitmapsetDynamic out = mask;
    for (; m < mask.nbits() && v < val.nbits(); m++) {
        if (mask.isSet(m)){
            if (!val.isSet(v))
                out.unset(m);
            v++;
        }
    }

    return out;
}


namespace std {
    template<>
    struct hash<BitmapsetDynamic> {
        size_t operator()(const BitmapsetDynamic& x) const {
            return x.hash();
        }
    };
}

#endif							/* GPUQO_BITMAPSET_DYNAMIC_CUH */
