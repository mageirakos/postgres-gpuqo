/*------------------------------------------------------------------------
 *
 * gpuqo_cost_postgres.cuh
 *      definition of the common cost-computing function (Postgres-like)
 *
 * src/backend/optimizer/gpuqo/gpuqo_cost_postgres.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_COST_POSTGRES_CUH
#define GPUQO_COST_POSTGRES_CUH

#include <cmath>
#include <cstdint>

#include "pg_config.h"

#include "optimizer/gpuqo_common.h"

#include "gpuqo.cuh"
#include "gpuqo_timing.cuh"
#include "gpuqo_debug.cuh"
#include "gpuqo_row_estimation.cuh"

#define SIZE_OF_HEAP_TUPLE_DATA sizeof(uintptr_t)
#define MaxAllocSize	((size_t) 0x3fffffff) /* 1 gigabyte - 1 */

#define TYPEALIGN(ALIGNVAL,LEN)  \
	(((uintptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
#define MAXALIGN(LEN) TYPEALIGN(MAXIMUM_ALIGNOF,LEN)

struct HashSkewBucketSkeleton
{
	uint32_t		hashvalue;
	void*           tuples;
};

#define LOG2(x)  (logf(x) / 0.693147180559945f)
#define LOG6(x)  (logf(x) / 1.79175946922805f)

#define COST_FUNCTION_OVERHEAD 3000L

struct QualCost{
    int n_quals;
    float startup;
    float per_tuple;
};

struct BucketStats {
	float mcvfreq;
	float bucketsize;
};

struct HtszRes {
	int numbuckets;
	int numbatches;
	int num_skew_mcvs;
};

__host__ __device__
static float relation_byte_size(float tuples, int width);

__host__ __device__
static float page_size(float tuples, int width);

__host__ __device__
static float clamp_row_est(float nrows);

__host__ __device__
static HtszRes ExecChooseHashTableSize(float ntuples, int tupwidth, int work_mem);



template <typename BitmapsetN>
__host__ __device__
static struct QualCost 
cost_qual_eval(BitmapsetN left_rel_id, BitmapsetN right_rel_id,
                GpuqoPlannerInfo<BitmapsetN>* info);

template <typename BitmapsetN>
__host__ __device__
static BucketStats 
estimate_hash_bucket_stats(BitmapsetN outer_rel_id, BitmapsetN inner_rel_id,
                            GpuqoPlannerInfo<BitmapsetN>* info);

template <typename BitmapsetN>
__host__ __device__
static float
__estimate_hash_bucketsize(VarInfo &stat, BaseRelation<BitmapsetN> &baserel, 
                        int nbuckets, GpuqoPlannerInfo<BitmapsetN>* info);


__host__ __device__
inline int floor_log2(long num) {
	return sizeof(num)*8 - 1 - clz(num);
}

__host__ __device__
inline int ceil_log2(long num) {
	int res = floor_log2(num);
	if (1L<<res == num)
		return res;
	else
		return res+1;
}

/*
 * cost_nestloop
 *	  Determines and returns the cost of joining two relations using the
 *	  nested loop algorithm.
 *
 * 'path' is already filled in except for the cost fields
 */
template <typename BitmapsetN>
__host__ __device__
static struct PathCost
cost_nestloop(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                CostExtra &extra, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		startup_cost = 0.0f;
	float		run_cost = 0.0f;
	float		cpu_per_tuple;
	float		inner_run_cost; 
	float		inner_rescan_run_cost;
	QualCost	restrict_qual_cost;
	float		outer_path_rows = outer_rel.rows;
	float		inner_path_rows = inner_rel.rows;
	float		ntuples;

	if (!info->params.enable_nestloop)
		startup_cost += info->params.disable_cost;

	/* cost of source data */

	/*
	 * NOTE: clearly, we must pay both outer and inner paths' startup_cost
	 * before we can start returning tuples, so the join's startup cost is
	 * their sum.  What's not so clear is whether the inner path's
	 * startup_cost must be paid again on each rescan of the inner path.
	 * This is not true if the inner path is materialized or is a
	 * hashjoin, but probably is true otherwise.
	 */
	startup_cost += inner_rel.cost.startup + outer_rel.cost.startup;
	run_cost += outer_rel.cost.total - outer_rel.cost.startup;
	if (outer_path_rows > 1)
		run_cost += (outer_path_rows - 1.0f) * inner_rel.cost.startup;

	inner_run_cost = inner_rel.cost.total - inner_rel.cost.startup;
	inner_rescan_run_cost = inner_run_cost;

	if (extra.inner_unique) {
		/*
		 * With a SEMI or ANTI join, or if the innerrel is known unique, the
		 * executor will stop after the first match.
		 */
		float		outer_matched_rows;
		float		outer_unmatched_rows;
		float		inner_scan_frac;

		float outer_match_frac = extra.joinrows / (inner_path_rows * outer_path_rows);
		float match_count = inner_path_rows;

		/*
		 * For an outer-rel row that has at least one match, we can expect the
		 * inner scan to stop after a fraction 1/(match_count+1) of the inner
		 * rows, if the matches are evenly distributed.  Since they probably
		 * aren't quite evenly distributed, we apply a fuzz factor of 2.0 to
		 * that fraction.  (If we used a larger fuzz factor, we'd have to
		 * clamp inner_scan_frac to at most 1.0; but since match_count is at
		 * least 1, no such clamp is needed now.)
		 */
		outer_matched_rows = rintf(outer_path_rows * outer_match_frac);
		outer_unmatched_rows = outer_path_rows - outer_matched_rows;
		inner_scan_frac = 2.0f / (match_count + 1.0f);

		/*
		 * Compute number of tuples processed (not number emitted!).  First,
		 * account for successfully-matched outer rows.
		 */
		ntuples = outer_matched_rows * inner_path_rows * inner_scan_frac;

		/*
		 * Now we need to estimate the actual costs of scanning the inner
		 * relation, which may be quite a bit less than N times inner_run_cost
		 * due to early scan stops.  We consider two cases.  If the inner path
		 * is an indexscan using all the joinquals as indexquals, then an
		 * unmatched outer row results in an indexscan returning no rows,
		 * which is probably quite cheap.  Otherwise, the executor will have
		 * to scan the whole inner rel for an unmatched row; not so cheap.
		 */
		if (extra.indexed_join_quals)
		{
			/*
			 * Successfully-matched outer rows will only require scanning
			 * inner_scan_frac of the inner relation.  In this case, we don't
			 * need to charge the full inner_run_cost even when that's more
			 * than inner_rescan_run_cost, because we can assume that none of
			 * the inner scans ever scan the whole inner relation.  So it's
			 * okay to assume that all the inner scan executions can be
			 * fractions of the full cost, even if materialization is reducing
			 * the rescan cost.  At this writing, it's impossible to get here
			 * for a materialized inner scan, so inner_run_cost and
			 * inner_rescan_run_cost will be the same anyway; but just in
			 * case, use inner_run_cost for the first matched tuple and
			 * inner_rescan_run_cost for additional ones.
			 */
			run_cost += inner_run_cost * inner_scan_frac;
			if (outer_matched_rows > 1)
				run_cost += (outer_matched_rows - 1) * inner_rescan_run_cost * inner_scan_frac;

			/*
			 * Add the cost of inner-scan executions for unmatched outer rows.
			 * We estimate this as the same cost as returning the first tuple
			 * of a nonempty scan.  We consider that these are all rescans,
			 * since we used inner_run_cost once already.
			 */
			run_cost += outer_unmatched_rows *
				inner_rescan_run_cost / inner_path_rows;

			/*
			 * We won't be evaluating any quals at all for unmatched rows, so
			 * don't add them to ntuples.
			 */
		}
		else
		{
			/*
			 * Here, a complicating factor is that rescans may be cheaper than
			 * first scans.  If we never scan all the way to the end of the
			 * inner rel, it might be (depending on the plan type) that we'd
			 * never pay the whole inner first-scan run cost.  However it is
			 * difficult to estimate whether that will happen (and it could
			 * not happen if there are any unmatched outer rows!), so be
			 * conservative and always charge the whole first-scan cost once.
			 * We consider this charge to correspond to the first unmatched
			 * outer row, unless there isn't one in our estimate, in which
			 * case blame it on the first matched row.
			 */

			/* First, count all unmatched join tuples as being processed */
			ntuples += outer_unmatched_rows * inner_path_rows;

			/* Now add the forced full scan, and decrement appropriate count */
			run_cost += inner_run_cost;
			if (outer_unmatched_rows >= 1)
				outer_unmatched_rows -= 1;
			else
				outer_matched_rows -= 1;

			/* Add inner run cost for additional outer tuples having matches */
			if (outer_matched_rows > 0)
				run_cost += outer_matched_rows * inner_rescan_run_cost * inner_scan_frac;

			/* Add inner run cost for additional unmatched outer tuples */
			if (outer_unmatched_rows > 0)
				run_cost += outer_unmatched_rows * inner_rescan_run_cost;
		}
	} else {
		/* Normal case; we'll scan whole input rel for each outer row */
		run_cost += inner_run_cost;
		if (outer_path_rows > 1)
			run_cost += (outer_path_rows - 1.0f) * inner_rescan_run_cost;

		/* Normal-case source costs were included in preliminary estimate */

		/* Compute number of tuples processed (not number emitted!) */
		ntuples = outer_path_rows * inner_path_rows;
	}

	/* CPU costs */
	restrict_qual_cost = cost_qual_eval(inner_rel_id, outer_rel_id, info);
	startup_cost += restrict_qual_cost.startup;
	cpu_per_tuple = info->params.cpu_tuple_cost + restrict_qual_cost.per_tuple;
	run_cost += cpu_per_tuple * ntuples;

	return (struct PathCost){
        .startup = startup_cost,
        .total = startup_cost + run_cost
    }; 
}

/*
 * cost_hashjoin
 *	  Refer to Postgres cost function for details.
 *
 * Assumptions:
 *  - all quals are of type "A = B"
 *  - inner join
 *  - not parallel
 * 
 */
template <typename BitmapsetN>
__host__ __device__
static struct PathCost 
cost_hashjoin(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                CostExtra extra, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		startup_cost = 0.0f;
	float		run_cost = 0.0f;
	float		cpu_per_tuple;
	QualCost	hash_qual_cost;
	float		hashjointuples;
	float		virtualbuckets;
	float		outer_path_rows = outer_rel.rows;
	float		inner_path_rows = inner_rel.rows;
	float		inner_path_rows_total = inner_path_rows;
	HtszRes htsz_res;
	BucketStats bucket_stats;

	if (!info->params.enable_hashjoin)
		startup_cost += info->params.disable_cost;

	
	/*
	 * Compute cost of the hashquals and qpquals (other restriction clauses)
	 * separately.
	 */
	hash_qual_cost = cost_qual_eval(outer_rel_id, inner_rel_id, info);

	/* cost of source data */
	startup_cost += outer_rel.cost.startup;
	run_cost += outer_rel.cost.total - outer_rel.cost.startup;
	startup_cost += inner_rel.cost.total;

	/*
	 * Cost of computing hash function: must do it once per input tuple. We
	 * charge one info->params.cpu_operator_cost for each column's hash function.  Also,
	 * tack on one info->params.cpu_tuple_cost per inner row, to model the costs of
	 * inserting the row into the hashtable.
	 *
	 * XXX when a hashclause is more complex than a single operator, we really
	 * should charge the extra eval costs of the left or right side, as
	 * appropriate, here.  This seems more work than it's worth at the moment.
	 */
	startup_cost += (info->params.cpu_operator_cost * hash_qual_cost.n_quals + info->params.cpu_tuple_cost)
		* inner_path_rows;
	run_cost += info->params.cpu_operator_cost * hash_qual_cost.n_quals * outer_path_rows;

	/*
	 * Get hash table size that executor would use for inner relation.
	 *
	 * XXX for the moment, always assume that skew optimization will be
	 * performed.  As long as SKEW_WORK_MEM_PERCENT is small, it's not worth
	 * trying to determine that for sure.
	 *
	 * XXX at some point it might be interesting to try to account for skew
	 * optimization in the cost estimate, but for now, we don't.
	 */
	htsz_res = ExecChooseHashTableSize(inner_path_rows_total, inner_rel.width, info->params.work_mem);

	/*
	 * If inner relation is too big then we will need to "batch" the join,
	 * which implies writing and reading most of the tuples to disk an extra
	 * time.  Charge seq_page_cost per page, since the I/O should be nice and
	 * sequential.  Writing the inner rel counts as startup cost, all the rest
	 * as run cost.
	 */
	if (htsz_res.numbatches > 1)
	{
		float		outerpages = page_size(outer_path_rows,
										   outer_rel.width);
		float		innerpages = page_size(inner_path_rows,
										   inner_rel.width);

		startup_cost += info->params.seq_page_cost * innerpages;
		run_cost += info->params.seq_page_cost * (innerpages + 2.0f * outerpages);
	}

	/* and compute the number of "virtual" buckets in the whole join */
	virtualbuckets = (float) htsz_res.numbuckets * (float) htsz_res.numbatches;


	bucket_stats = estimate_hash_bucket_stats(outer_rel_id, inner_rel_id, virtualbuckets, info);

	/*
	 * If the bucket holding the inner MCV would exceed work_mem, we don't
	 * want to hash unless there is really no other alternative, so apply
	 * disable_cost.  (The executor normally copes with excessive memory usage
	 * by splitting batches, but obviously it cannot separate equal values
	 * that way, so it will be unable to drive the batch size below work_mem
	 * when this is true.)
	 */
	if (relation_byte_size(clamp_row_est(inner_path_rows * bucket_stats.mcvfreq),
						   inner_rel.width) >
		(info->params.work_mem * 1024L))
		startup_cost += info->params.disable_cost;

	/* CPU costs */

	if (extra.inner_unique)
	{
		float		outer_matched_rows;
		float 		inner_scan_frac;

		float outer_match_frac = extra.joinrows / (inner_path_rows * outer_path_rows);
		float match_count = inner_path_rows;

		/*
		 * With a SEMI or ANTI join, or if the innerrel is known unique, the
		 * executor will stop after the first match.
		 *
		 * For an outer-rel row that has at least one match, we can expect the
		 * bucket scan to stop after a fraction 1/(match_count+1) of the
		 * bucket's rows, if the matches are evenly distributed.  Since they
		 * probably aren't quite evenly distributed, we apply a fuzz factor of
		 * 2.0 to that fraction.  (If we used a larger fuzz factor, we'd have
		 * to clamp inner_scan_frac to at most 1.0; but since match_count is
		 * at least 1, no such clamp is needed now.)
		 */
		outer_matched_rows = rintf(outer_path_rows * outer_match_frac);
		inner_scan_frac = 2.0f / (match_count + 1.0f);

		startup_cost += hash_qual_cost.startup;
		run_cost += hash_qual_cost.per_tuple * outer_matched_rows *
			clamp_row_est(inner_path_rows * bucket_stats.bucketsize * inner_scan_frac) * 0.5f;

		/*
		 * For unmatched outer-rel rows, the picture is quite a lot different.
		 * In the first place, there is no reason to assume that these rows
		 * preferentially hit heavily-populated buckets; instead assume they
		 * are uncorrelated with the inner distribution and so they see an
		 * average bucket size of inner_path_rows / virtualbuckets.  In the
		 * second place, it seems likely that they will have few if any exact
		 * hash-code matches and so very few of the tuples in the bucket will
		 * actually require eval of the hash quals.  We don't have any good
		 * way to estimate how many will, but for the moment assume that the
		 * effective cost per bucket entry is one-tenth what it is for
		 * matchable tuples.
		 */
		run_cost += hash_qual_cost.per_tuple *
			(outer_path_rows - outer_matched_rows) *
			clamp_row_est(inner_path_rows / virtualbuckets) * 0.05f;

		/* Get # of tuples that will pass the basic join */
		hashjointuples = outer_matched_rows;
	}
	else
	{
		/*
		* The number of tuple comparisons needed is the number of outer
		* tuples times the typical number of tuples in a hash bucket, which
		* is the inner relation size times its bucketsize fraction.  At each
		* one, we need to evaluate the hashjoin quals.  But actually,
		* charging the full qual eval cost at each tuple is pessimistic,
		* since we don't evaluate the quals unless the hash values match
		* exactly.  For lack of a better idea, halve the cost estimate to
		* allow for that.
		*/
		startup_cost += hash_qual_cost.startup;
		run_cost += hash_qual_cost.per_tuple * outer_path_rows *
			clamp_row_est(inner_path_rows * bucket_stats.bucketsize) * 0.5f;

		/*
		* Get approx # tuples passing the hashquals.  We use
		* approx_tuple_count here because we need an estimate done with
		* JOIN_INNER semantics.
		*/
		hashjointuples = extra.joinrows;
	}

	/*
	 * For each tuple that gets through the hashjoin proper, we charge
	 * cpu_tuple_cost plus the cost of evaluating additional restriction
	 * clauses that are to be applied at the join.  (This is pessimistic since
	 * not all of the quals may get evaluated at each tuple.)
	 */
	cpu_per_tuple = info->params.cpu_tuple_cost;
	run_cost += cpu_per_tuple * hashjointuples;

	/* tlist WTF? */

	return (struct PathCost){
        .startup = startup_cost,
        .total = startup_cost + run_cost
    }; 
}

/* Magic numbers (see Postgres) */
#define NTUP_PER_BUCKET			1
#define SKEW_WORK_MEM_PERCENT	2
#define SKEW_OVERHEAD  84
#define TUPLE_OVERHEAD  32
#define RELATION_OVERHEAD 24

HtszRes
ExecChooseHashTableSize(float ntuples, int tupwidth, int work_mem)
{
	int			tupsize;
	float		inner_rel_bytes;
	long		bucket_bytes;
	long		hash_table_bytes;
	long		skew_table_bytes;
	long		max_pointers;
	long		mppow2;
	float		dbuckets;
	float		space_allowed;
	HtszRes     res;

	res.numbatches = 1;

	/* Force a plausible relation size if no info */
	if (ntuples <= 0.0)
		ntuples = 1000.0;

	/*
	 * Estimate tupsize based on footprint of tuple in hashtable... note this
	 * does not allow for any palloc overhead.  The manipulations of spaceUsed
	 * don't count palloc overhead either.
	 */
	tupsize = TUPLE_OVERHEAD + MAXALIGN(tupwidth);
	inner_rel_bytes = ntuples * tupsize;

	/*
	 * Target in-memory hashtable size is work_mem kilobytes.
	 */
	hash_table_bytes = work_mem * 1024L;

	space_allowed = hash_table_bytes;

	/*
	 * If skew optimization is possible, estimate the number of skew buckets
	 * that will fit in the memory allowed, and decrement the assumed space
	 * available for the main hash table accordingly.
	 *
	 * We make the optimistic assumption that each skew bucket will contain
	 * one inner-relation tuple.  If that turns out to be low, we will recover
	 * at runtime by reducing the number of skew buckets.
	 *
	 * hashtable->skewBucket will have up to 8 times as many HashSkewBucket
	 * pointers as the number of MCVs we allow, since ExecHashBuildSkewHash
	 * will round up to the next power of 2 and then multiply by 4 to reduce
	 * collisions.
	 */

	skew_table_bytes = hash_table_bytes * SKEW_WORK_MEM_PERCENT / 100;

	/*----------
		* Divisor is:
		* size of a hash tuple +
		* worst-case size of skewBucket[] per MCV +
		* size of skewBucketNums[] entry +
		* size of skew bucket struct itself
		*----------
		*/
	res.num_skew_mcvs = skew_table_bytes / (tupsize + SKEW_OVERHEAD);
	if (res.num_skew_mcvs > 0)
		hash_table_bytes -= skew_table_bytes;

	/*
	 * Set nbuckets to achieve an average bucket load of NTUP_PER_BUCKET when
	 * memory is filled, assuming a single batch; but limit the value so that
	 * the pointer arrays we'll try to allocate do not exceed work_mem nor
	 * MaxAllocSize.
	 *
	 * Note that both nbuckets and nbatch must be powers of 2 to make
	 * ExecHashGetBucketAndBatch fast.
	 */
	max_pointers = space_allowed / sizeof(void*);
	max_pointers = min(max_pointers, MaxAllocSize / sizeof(void*));
	/* If max_pointers isn't a power of 2, must round it down to one */
	mppow2 = 1L << ceil_log2(max_pointers);
	if (max_pointers != mppow2)
		max_pointers = mppow2 / 2;

	/* Also ensure we avoid integer overflow in nbatch and nbuckets */
	/* (this step is redundant given the current value of MaxAllocSize) */
	max_pointers = min(max_pointers, (long) INT_MAX / 2);

	dbuckets = ceilf(ntuples / NTUP_PER_BUCKET);
	dbuckets = min(dbuckets, (float) max_pointers);
	res.numbuckets  = (int) dbuckets;
	/* don't let nbuckets be really small, though ... */
	res.numbuckets  = max(res.numbuckets , 1024);
	/* ... and force it to be a power of 2. */
	res.numbuckets  = 1 << ceil_log2(res.numbuckets );

	/*
	 * If there's not enough space to store the projected number of tuples and
	 * the required bucket headers, we will need multiple batches.
	 */
	bucket_bytes = sizeof(void*) * res.numbuckets ;
	if (inner_rel_bytes + bucket_bytes > hash_table_bytes)
	{
		/* We'll need multiple batches */
		long		lbuckets;
		float		dbatch;
		int			minbatch;
		long		bucket_size;

		/*
		 * Estimate the number of buckets we'll want to have when work_mem is
		 * entirely full.  Each bucket will contain a bucket pointer plus
		 * NTUP_PER_BUCKET tuples, whose projected size already includes
		 * overhead for the hash code, pointer to the next tuple, etc.
		 */
		bucket_size = (tupsize * NTUP_PER_BUCKET + sizeof(void*));
		lbuckets = 1L << ceil_log2(hash_table_bytes / bucket_size);
		lbuckets = min(lbuckets, max_pointers);
		res.numbuckets = (int) lbuckets;
		res.numbuckets = 1 << ceil_log2(res.numbuckets);
		bucket_bytes = res.numbuckets * sizeof(void*);

		/*
		 * Buckets are simple pointers to hashjoin tuples, while tupsize
		 * includes the pointer, hash code, and MinimalTupleData.  So buckets
		 * should never really exceed 25% of work_mem (even for
		 * NTUP_PER_BUCKET=1); except maybe for work_mem values that are not
		 * 2^N bytes, where we might get more because of doubling. So let's
		 * look for 50% here.
		 */
		Assert(bucket_bytes <= hash_table_bytes / 2);

		/* Calculate required number of batches. */
		dbatch = ceilf(inner_rel_bytes / (hash_table_bytes - bucket_bytes));
		dbatch = min(dbatch, (float) max_pointers);
		minbatch = (int) dbatch;
		res.numbatches = max(2, 1<<ceil_log2(minbatch));
	}

	Assert(res.numbuckets > 0);
	Assert(res.numbatches > 0);

	return res;
}

template <typename BitmapsetN>
__host__ __device__
static BucketStats 
estimate_hash_bucket_stats(BitmapsetN outer_rel_id, BitmapsetN inner_rel_id,
                            int nbuckets,
                            GpuqoPlannerInfo<BitmapsetN>* info)
{
    BucketStats stats;
	
	stats.bucketsize = 1.0f;
	stats.mcvfreq = 1.0f;

    // for each ec that involves any baserel on the left and on the right,
    // count 1 cpu operation (we are assuming 'equals' operators only)
    for (int i=0, off_vars=0; 
		i<info->eq_classes.n; 
		off_vars += info->eq_classes.relids[i].size(),
		i++)
	{
        BitmapsetN ec_relids = info->eq_classes.relids[i];
        
        BitmapsetN match_l = ec_relids & outer_rel_id;
        BitmapsetN match_r = ec_relids & inner_rel_id;

        if (match_l.empty() || match_r.empty())
            continue;

        while(!match_r.empty()){
            BitmapsetN in_id = match_r.lowest();
            int in_idx = (in_id.allLower() & ec_relids).size();

            BaseRelation<BitmapsetN>& baserel = info->base_rels[in_id.lowestPos()-1];
            VarInfo var = info->eq_classes.vars[off_vars+in_idx];

            float thisbucketsize = __estimate_hash_bucketsize(var, baserel, nbuckets, info);
            
            if (stats.bucketsize > thisbucketsize)
                stats.bucketsize = thisbucketsize;

            if (stats.mcvfreq > var.stats.mcvfreq)
                stats.mcvfreq = var.stats.mcvfreq;

            match_r ^= in_id;
        }
    }
    
    return stats;
}


/*
 * Estimate hash bucketsize fraction (ie, number of entries in a bucket
 * divided by total tuples in relation) if the specified Var is used
 * as a hash key.
 *
 * XXX This is really pretty bogus since we're effectively assuming that the
 * distribution of hash keys will be the same after applying restriction
 * clauses as it was in the underlying relation.  However, we are not nearly
 * smart enough to figure out how the restrict clauses might change the
 * distribution, so this will have to do for now.
 *
 * We are passed the number of buckets the executor will use for the given
 * input relation.	If the data were perfectly distributed, with the same
 * number of tuples going into each available bucket, then the bucketsize
 * fraction would be 1/nbuckets.  But this happy state of affairs will occur
 * only if (a) there are at least nbuckets distinct data values, and (b)
 * we have a not-too-skewed data distribution.	Otherwise the buckets will
 * be nonuniformly occupied.  If the other relation in the join has a key
 * distribution similar to this one's, then the most-loaded buckets are
 * exactly those that will be probed most often.  Therefore, the "average"
 * bucket size for costing purposes should really be taken as something close
 * to the "worst case" bucket size.  We try to estimate this by adjusting the
 * fraction if there are too few distinct data values, and then scaling up
 * by the ratio of the most common value's frequency to the average frequency.
 *
 * If no statistics are available, use a default estimate of 0.1.  This will
 * discourage use of a hash rather strongly if the inner relation is large,
 * which is what we want.  We do not want to hash unless we know that the
 * inner rel is well-dispersed (or the alternatives seem much worse).
 */
template <typename BitmapsetN>
__host__ __device__
static float
__estimate_hash_bucketsize(VarInfo &vars, BaseRelation<BitmapsetN> &baserel, 
                        int nbuckets, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		estfract,
				ndistinct,
				mcvfreq,
				avgfreq;

	/*
	 * Obtain number of distinct data values in raw relation.
	 */
	ndistinct = vars.stats.stadistinct;
	if (ndistinct < 0.0f)
		ndistinct = -ndistinct * baserel.tuples;

	if (ndistinct <= 0.0f)		/* ensure we can divide */
	{
		return 0.1f;
	}

	/* Also compute avg freq of all distinct data values in raw relation */
	avgfreq = (1.0f - vars.stats.stanullfrac) / ndistinct;

	/*
	 * Adjust ndistinct to account for restriction clauses.  Observe we are
	 * assuming that the data distribution is affected uniformly by the
	 * restriction clauses!
	 *
	 * XXX Possibly better way, but much more expensive: multiply by
	 * selectivity of rel's restriction clauses that mention the target Var.
	 */
	if (baserel.tuples > 0)
	{
		ndistinct *= baserel.rows / baserel.tuples;
		ndistinct = clamp_row_est(ndistinct);
	}

	/*
	 * Initial estimate of bucketsize fraction is 1/nbuckets as long as
	 * the number of buckets is less than the expected number of distinct
	 * values; otherwise it is 1/ndistinct.
	 */
	if (ndistinct > nbuckets)
		estfract = 1.0f / nbuckets;
	else
		estfract = 1.0f / ndistinct;

	/*
	 * Look up the frequency of the most common value, if available.
	 */
	mcvfreq = vars.stats.mcvfreq;

	/*
	 * Adjust estimated bucketsize upward to account for skewed
	 * distribution.
	 */
	if (avgfreq > 0.0f && mcvfreq > avgfreq)
		estfract *= mcvfreq / avgfreq;

	/*
	 * Clamp bucketsize to sane range (the above adjustment could easily
	 * produce an out-of-range result).  We set the lower bound a little
	 * above zero, since zero isn't a very sane result.
	 */
	if (estfract < 1.0e-6f)
		estfract = 1.0e-6f;
	else if (estfract > 1.0f)
		estfract = 1.0f;

	return estfract;
}


/*
 * cost_qual_eval
 *		Estimate the CPU costs of evaluating a WHERE clause.
 *		The input can be either an implicitly-ANDed list of boolean
 *		expressions, or a list of RestrictInfo nodes.
 *		The result includes both a one-time (startup) component,
 *		and a per-evaluation component.
 */
template <typename BitmapsetN>
__host__ __device__
static struct QualCost 
cost_qual_eval(BitmapsetN left_rel_id, BitmapsetN right_rel_id,
                GpuqoPlannerInfo<BitmapsetN>* info)
{
    struct QualCost cost;

    cost.startup = 0.0f;
    cost.per_tuple = 0.0f;
    cost.n_quals = 0.0f;

    // for each ec that involves any baserel on the left and on the right,
    // count 1 cpu operation (we are assuming 'equals' operators only)
    for (int i=0; i<info->eq_classes.n; i++){
        BitmapsetN ec_relids = info->eq_classes.relids[i];
        
        BitmapsetN match_l = ec_relids & left_rel_id;
        BitmapsetN match_r = ec_relids & right_rel_id;

        if (match_l.empty() || match_r.empty())
            continue;

        cost.per_tuple += info->params.cpu_operator_cost;
        cost.n_quals++;
    }
    
    return cost;
}

/*
 * relation_byte_size
 *	  Estimate the storage space in bytes for a given number of tuples
 *	  of a given width (size in bytes).
 */
__host__ __device__
static float
relation_byte_size(float tuples, int width)
{
	return tuples * (MAXALIGN(width) + RELATION_OVERHEAD);
}

/*
 * page_size
 *	  Returns an estimate of the number of pages covered by a given
 *	  number of tuples of a given width (size in bytes).
 */
__host__ __device__
static float
page_size(float tuples, int width)
{
	return ceil(relation_byte_size(tuples, width) / BLCKSZ);
}

template<typename BitmapsetN>
__host__ __device__
static struct PathCost
cost_baserel(BaseRelation<BitmapsetN> &base_rel){
    return base_rel.cost;
}

static float 
clamp_row_est(float nrows)
{
	/*
	 * Force estimate to be at least one row, to make explain output look
	 * better and to avoid possible divide-by-zero when interpolating costs.
	 * Make it an integer, too.
	 */
	if (nrows <= 1.0)
		nrows = 1.0;
	else
		nrows = rintf(nrows);

	return nrows;
}

#endif
