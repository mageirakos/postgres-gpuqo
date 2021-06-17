/*-------------------------------------------------------------------------
 *
 * gpuqo_postgres.cuh
 *	  postgres bindings for C++/CUDA
 *
 * src/include/optimizer/gpuqo_postgres.cuh
 *
 *-------------------------------------------------------------------------
 */
#ifndef GPUQO_POSTGRES_CUH
#define GPUQO_POSTGRES_CUH

#define BITMAPSET_H // do not include bitmapsets

extern "C" void *palloc0(size_t size);
extern "C" void *palloc(size_t size);

typedef uint64_t uint64;
typedef int64_t int64;
typedef uint32_t uint32;
typedef int32_t int32;

typedef uint64 bitmapword;
#define BITS_PER_BITMAPWORD 64

#define SIZEOF_VOID_P 8
#define FLEXIBLE_ARRAY_MEMBER /**/

struct Bitmapset
{
	int			nwords;
	bitmapword	words[FLEXIBLE_ARRAY_MEMBER];
};

extern "C" Bitmapset *bms_copy(const Bitmapset *a);
extern "C" bool bms_equal(const Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_make_singleton(int x);
extern "C" void bms_free(Bitmapset *a);

extern "C" Bitmapset *bms_union(const Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_intersect(const Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_difference(const Bitmapset *a, const Bitmapset *b);
extern "C" bool bms_is_subset(const Bitmapset *a, const Bitmapset *b);
extern "C" bool bms_is_member(int x, const Bitmapset *a);
extern "C" bool bms_overlap(const Bitmapset *a, const Bitmapset *b);
extern "C" int	bms_num_members(const Bitmapset *a);

/* optimized tests when we don't need to know exact membership count: */
extern "C" bool bms_is_empty(const Bitmapset *a);

/* these routines recycle (modify or free) their non-const inputs: */

extern "C" Bitmapset *bms_add_member(Bitmapset *a, int x);
extern "C" Bitmapset *bms_del_member(Bitmapset *a, int x);
extern "C" Bitmapset *bms_add_members(Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_add_range(Bitmapset *a, int lower, int upper);
extern "C" Bitmapset *bms_int_members(Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_del_members(Bitmapset *a, const Bitmapset *b);
extern "C" Bitmapset *bms_join(Bitmapset *a, Bitmapset *b);

/* support for hashtables using Bitmapsets as keys: */
extern "C" uint32 bms_hash_value(const Bitmapset *a);

#include <optimizer/gpuqo_planner_info.h>

#endif							/* GPUQO_POSTGRES_CUH */
