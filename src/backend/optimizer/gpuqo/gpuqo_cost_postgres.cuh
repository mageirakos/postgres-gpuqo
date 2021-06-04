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

#define LOG2(x)  (logf(x) / 0.693147180559945f)
#define LOG6(x)  (logf(x) / 1.79175946922805f)

#define COST_FUNCTION_OVERHEAD 3000L

struct QualCost{
    int n_quals;
    float startup;
    float per_tuple;
};

__host__ __device__
static float relation_byte_size(float tuples, int width);

__host__ __device__
static float page_size(float tuples, int width);

__host__ __device__
static void ExecChooseHashTableSize(float ntuples, int tupwidth, int sort_mem,
                                    int *virtualbuckets,
                                    int *physicalbuckets,
                                    int *numbatches);

template <typename BitmapsetN>
__host__ __device__
static struct QualCost 
cost_qual_eval(BitmapsetN left_rel_id, BitmapsetN right_rel_id,
                GpuqoPlannerInfo<BitmapsetN>* info);

template <typename BitmapsetN>
__host__ __device__
static float 
estimate_hash_innerbucketsize(BitmapsetN outer_rel_id, BitmapsetN inner_rel_id,
                            GpuqoPlannerInfo<BitmapsetN>* info);

template <typename BitmapsetN>
__host__ __device__
static float
__estimate_hash_bucketsize(VarStat &stat, BaseRelation<BitmapsetN> &baserel, 
                        int nbuckets, GpuqoPlannerInfo<BitmapsetN>* info);

/*
 * cost_nestloop
 *	  Determines and returns the cost of joining two relations using the
 *	  nested loop algorithm.
 *
 * 'path' is already filled in except for the cost fields
 */
template <typename BitmapsetN>
__host__ __device__
static struct Cost
cost_nestloop(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		startup_cost = 0.0f;
	float		run_cost = 0.0f;
	float		cpu_per_tuple;
	QualCost	restrict_qual_cost;
	float		outer_path_rows = outer_rel.rows;
	float		inner_path_rows = inner_rel.rows;
	float		ntuples;
	float       joininfactor;

	if (!info->params.enable_nestloop)
		startup_cost += info->params.disable_cost;

	/*
	 * If we're doing JOIN_IN then we will stop scanning inner tuples for
	 * an outer tuple as soon as we have one match.  Account for the
	 * effects of this by scaling down the cost estimates in proportion to
	 * the expected output size.  (This assumes that all the quals
	 * attached to the join are IN quals, which should be true.)
	 *
	 * Note: it's probably bogus to use the normal selectivity calculation
	 * here when either the outer or inner path is a UniquePath.
	 */
    // TODO check if this is right
    // in our case, all quals are in the join so the selectivity should be the 
    // same    
    joininfactor = 1.0f;

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

    /*
        * charge startup cost for each iteration of inner path, except we
        * already charged the first startup_cost in our own startup
    */
    run_cost += (outer_path_rows - 1.0f) * inner_rel.cost.startup;

	run_cost += outer_path_rows *
		(inner_rel.cost.total - inner_rel.cost.startup) * joininfactor;

	ntuples = inner_path_rows * outer_path_rows;

	/* CPU costs */
	restrict_qual_cost = cost_qual_eval(inner_rel_id, outer_rel_id, info);
	startup_cost += restrict_qual_cost.startup;
	cpu_per_tuple = info->params.cpu_tuple_cost + restrict_qual_cost.per_tuple;
	run_cost += cpu_per_tuple * ntuples;

	return (struct Cost){
        .startup = startup_cost,
        .total = startup_cost + run_cost
    }; 
}

/*
 * cost_hashjoin
 *	  Determines and returns the cost of joining two relations using the
 *	  hash join algorithm.
 *
 * 'path' is already filled in except for the cost fields
 *
 * Note: path's hashclauses should be a subset of the joinrestrictinfo list
 */
template <typename BitmapsetN>
__host__ __device__
static struct Cost 
cost_hashjoin(BitmapsetN outer_rel_id, JoinRelation<BitmapsetN> &outer_rel,
                BitmapsetN inner_rel_id, JoinRelation<BitmapsetN> &inner_rel,
                float join_rel_rows, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		startup_cost = 0.0f;
	float		run_cost = 0.0f;
	float		cpu_per_tuple;
	float       hash_selec;
	float       qp_selec;
	QualCost	hash_qual_cost;
	QualCost	qp_qual_cost;
	float		hashjointuples;
	float		__attribute__((unused)) qptuples; // TODO
	float		outer_path_rows = outer_rel.rows;
	float		inner_path_rows = inner_rel.rows;
	float		outerbytes = relation_byte_size(outer_path_rows,
											  outer_rel.width);
	float		innerbytes = relation_byte_size(inner_path_rows,
											  outer_rel.width);
	int			num_hashclauses;
	int			virtualbuckets;
	int			physicalbuckets;
	int			numbatches;
	float innerbucketsize;
	float joininfactor;

	if (!info->params.enable_hashjoin)
		startup_cost += info->params.disable_cost;

	/*
	 * Compute cost and selectivity of the hashquals and qpquals (other
	 * restriction clauses) separately.  We use approx_selectivity here
	 * for speed --- in most cases, any errors won't affect the result
	 * much.
	 *
	 * Note: it's probably bogus to use the normal selectivity calculation
	 * here when either the outer or inner path is a UniquePath.
	 */
    // TODO check
    // I am assuming all clauses are hashclauses
	hash_selec = min(1.0f, join_rel_rows/(outer_path_rows*inner_path_rows));
	hash_qual_cost = cost_qual_eval(outer_rel_id, inner_rel_id, info);
	num_hashclauses = hash_qual_cost.n_quals; // TODO check
	qp_selec = 1.0f;
	qp_qual_cost.startup = 0.0f;
	qp_qual_cost.per_tuple = 0.0f;

	/* approx # tuples passing the hash quals */
	hashjointuples = ceil(hash_selec * outer_path_rows * inner_path_rows);
	/* approx # tuples passing qpquals as well */
	qptuples = ceil(hashjointuples * qp_selec);

	/* cost of source data */
	startup_cost += outer_rel.cost.startup;
	run_cost += outer_rel.cost.total - outer_rel.cost.startup;
	startup_cost += inner_rel.cost.total;

	/*
	 * Cost of computing hash function: must do it once per input tuple.
	 * We charge one cpu_operator_cost for each column's hash function.
	 *
	 * XXX when a hashclause is more complex than a single operator, we
	 * really should charge the extra eval costs of the left or right
	 * side, as appropriate, here.	This seems more work than it's worth
	 * at the moment.
	 */
	startup_cost += info->params.cpu_operator_cost * num_hashclauses * inner_path_rows;
	run_cost += info->params.cpu_operator_cost * num_hashclauses * outer_path_rows;

	/* Get hash table size that executor would use for inner relation */
	ExecChooseHashTableSize(inner_path_rows,
							inner_rel.width,
                            info->params.work_mem,
							&virtualbuckets,
							&physicalbuckets,
							&numbatches);

	/*
	 * Determine bucketsize fraction for inner relation.  We use the
	 * smallest bucketsize estimated for any individual hashclause; this
	 * is undoubtedly conservative.
	 *
	 * BUT: if inner relation has been unique-ified, we can assume it's good
	 * for hashing.  This is important both because it's the right answer,
	 * and because we avoid contaminating the cache with a value that's
	 * wrong for non-unique-ified paths.
	 */
    // assuming not unique
    innerbucketsize = estimate_hash_innerbucketsize(outer_rel_id, inner_rel_id, virtualbuckets, info);

	/*
	 * if inner relation is too big then we will need to "batch" the join,
	 * which implies writing and reading most of the tuples to disk an
	 * extra time.	Charge one cost unit per page of I/O (correct since it
	 * should be nice and sequential...).  Writing the inner rel counts as
	 * startup cost, all the rest as run cost.
	 */
	if (numbatches)
	{
		float		outerpages = page_size(outer_path_rows,
										   outer_rel.width);
		float		innerpages = page_size(inner_path_rows,
										   inner_rel.width);

		startup_cost += innerpages;
		run_cost += innerpages + 2.0f * outerpages;
	}

	/* CPU costs */

	/*
	 * If we're doing JOIN_IN then we will stop comparing inner tuples to
	 * an outer tuple as soon as we have one match.  Account for the
	 * effects of this by scaling down the cost estimates in proportion to
	 * the expected output size.  (This assumes that all the quals
	 * attached to the join are IN quals, which should be true.)
	 */
    // assuming all clauses are hash clauses
    joininfactor = 1.0f;

	/*
	 * The number of tuple comparisons needed is the number of outer
	 * tuples times the typical number of tuples in a hash bucket, which
	 * is the inner relation size times its bucketsize fraction.  At each
	 * one, we need to evaluate the hashjoin quals.
	 */
	startup_cost += hash_qual_cost.startup;
	run_cost += hash_qual_cost.per_tuple *
		outer_path_rows * ceil(inner_path_rows * innerbucketsize) *
		joininfactor;

	/*
	 * For each tuple that gets through the hashjoin proper, we charge
	 * cpu_tuple_cost plus the cost of evaluating additional restriction
	 * clauses that are to be applied at the join.	(This is pessimistic
	 * since not all of the quals may get evaluated at each tuple.)
	 */
	startup_cost += qp_qual_cost.startup;
	cpu_per_tuple = info->params.cpu_tuple_cost + qp_qual_cost.per_tuple;
	run_cost += cpu_per_tuple * hashjointuples * joininfactor;

	/*
	 * Bias against putting larger relation on inside.	We don't want an
	 * absolute prohibition, though, since larger relation might have
	 * better bucketsize --- and we can't trust the size estimates
	 * unreservedly, anyway.  Instead, inflate the run cost by the square
	 * root of the size ratio.	(Why square root?  No real good reason,
	 * but it seems reasonable...)
	 *
	 * Note: before 7.4 we implemented this by inflating startup cost; but if
	 * there's a disable_cost component in the input paths' startup cost,
	 * that unfairly penalizes the hash.  Probably it'd be better to keep
	 * track of disable penalty separately from cost.
	 */
	if (innerbytes > outerbytes && outerbytes > 0.0f)
		run_cost *= sqrtf(innerbytes / outerbytes);

	return (struct Cost){
        .startup = startup_cost,
        .total = startup_cost + run_cost
    }; 
}

/* Target bucket loading (tuples per bucket) */
#define NTUP_PER_BUCKET			10
/* Fudge factor to allow for inaccuracy of input estimates */
#define FUDGE_FAC				2.0

__host__ __device__
static void
ExecChooseHashTableSize(float ntuples, int tupwidth, int sort_mem,
						int *virtualbuckets,
						int *physicalbuckets,
						int *numbatches)
{
	int			tupsize;
	float		inner_rel_bytes;
	long		hash_table_bytes;
	float		dtmp;
	int			nbatch;
	int			nbuckets;
	int			totalbuckets;
	int			bucketsize;

	/* Force a plausible relation size if no info */
	if (ntuples <= 0.0f)
		ntuples = 1000.0f;

	/*
	 * Estimate tupsize based on footprint of tuple in hashtable... but
	 * what about palloc overhead?
	 */
	tupsize = MAXALIGN(tupwidth) + MAXALIGN(sizeof(void*));
	inner_rel_bytes = ntuples * tupsize * FUDGE_FAC;

	/*
	 * Target in-memory hashtable size is sort_mem kilobytes.
	 */
	hash_table_bytes = sort_mem * 1024L;

	/*
	 * Count the number of hash buckets we want for the whole relation,
	 * for an average bucket load of NTUP_PER_BUCKET (per virtual
	 * bucket!).  It has to fit in an int, however.
	 */
	dtmp = ceil(ntuples * FUDGE_FAC / NTUP_PER_BUCKET);
	if (dtmp < INT_MAX)
		totalbuckets = (int) dtmp;
	else
		totalbuckets = INT_MAX;
	if (totalbuckets <= 0)
		totalbuckets = 1;

	/*
	 * Count the number of buckets we think will actually fit in the
	 * target memory size, at a loading of NTUP_PER_BUCKET (physical
	 * buckets). NOTE: FUDGE_FAC here determines the fraction of the
	 * hashtable space reserved to allow for nonuniform distribution of
	 * hash values. Perhaps this should be a different number from the
	 * other uses of FUDGE_FAC, but since we have no real good way to pick
	 * either one...
	 */
	bucketsize = NTUP_PER_BUCKET * tupsize;
	nbuckets = (int) (hash_table_bytes / (bucketsize * FUDGE_FAC));
	if (nbuckets <= 0)
		nbuckets = 1;
	/* Ensure we can allocate an array of nbuckets pointers */
	nbuckets = min(nbuckets, (int)(MaxAllocSize / sizeof(void *)));

	if (totalbuckets <= nbuckets)
	{
		/*
		 * We have enough space, so no batching.  In theory we could even
		 * reduce nbuckets, but since that could lead to poor behavior if
		 * estimated ntuples is much less than reality, it seems better to
		 * make more buckets instead of fewer.
		 */
		totalbuckets = nbuckets;
		nbatch = 0;
	}
	else
	{
		/*
		 * Need to batch; compute how many batches we want to use. Note
		 * that nbatch doesn't have to have anything to do with the ratio
		 * totalbuckets/nbuckets; in fact, it is the number of groups we
		 * will use for the part of the data that doesn't fall into the
		 * first nbuckets hash buckets.  We try to set it to make all the
		 * batches the same size.
		 */
		dtmp = ceil((inner_rel_bytes - hash_table_bytes) /
					hash_table_bytes);
		if (dtmp < MaxAllocSize / sizeof(void *))
			nbatch = (int) dtmp;
		else
			nbatch = MaxAllocSize / sizeof(void *);
		if (nbatch <= 0)
			nbatch = 1;
	}

	/*
	 * Now, totalbuckets is the number of (virtual) hashbuckets for the
	 * whole relation, and nbuckets is the number of physical hashbuckets
	 * we will use in the first pass.  Data falling into the first
	 * nbuckets virtual hashbuckets gets handled in the first pass;
	 * everything else gets divided into nbatch batches to be processed in
	 * additional passes.
	 */
	*virtualbuckets = totalbuckets;
	*physicalbuckets = nbuckets;
	*numbatches = nbatch;
}

template <typename BitmapsetN>
__host__ __device__
static float 
estimate_hash_innerbucketsize(BitmapsetN outer_rel_id, BitmapsetN inner_rel_id,
                            int nbuckets,
                            GpuqoPlannerInfo<BitmapsetN>* info)
{
    float innerbucketsize = 1.0f;

    // for each ec that involves any baserel on the left and on the right,
    // count 1 cpu operation (we are assuming 'equals' operators only)
    int off_stats = 0;
    for (int i=0; i<info->eq_classes.n; i++){
        BitmapsetN ec_relids = info->eq_classes.relids[i];
        
        BitmapsetN match_l = ec_relids & outer_rel_id;
        BitmapsetN match_r = ec_relids & inner_rel_id;

        if (match_l.empty() || match_r.empty())
            continue;

        while(!match_l.empty()){
            BitmapsetN out_id = match_l.lowest();
            int out_idx = (out_id.allLower() & ec_relids).size();

            BaseRelation<BitmapsetN>& baserel = info->base_rels[out_id.lowestPos()-1];
            VarStat stat = info->eq_classes.stats[off_stats+out_idx];

            float thisbucketsize = __estimate_hash_bucketsize(stat, baserel, nbuckets, info);
            
            if (innerbucketsize > thisbucketsize)
                innerbucketsize = thisbucketsize;

            match_l ^= out_id;
        }

        while(!match_r.empty()){
            BitmapsetN in_id = match_r.lowest();
            int in_idx = (in_id.allLower() & ec_relids).size();

            BaseRelation<BitmapsetN>& baserel = info->base_rels[in_id.lowestPos()-1];
            VarStat stat = info->eq_classes.stats[off_stats+in_idx];

            float thisbucketsize = __estimate_hash_bucketsize(stat, baserel, nbuckets, info);
            
            if (innerbucketsize > thisbucketsize)
                innerbucketsize = thisbucketsize;

            match_r ^= in_id;
        }

        off_stats += ec_relids.size();
    }
    
    return innerbucketsize;
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
__estimate_hash_bucketsize(VarStat &stats, BaseRelation<BitmapsetN> &baserel, 
                        int nbuckets, GpuqoPlannerInfo<BitmapsetN>* info)
{
	float		estfract,
				ndistinct,
				mcvfreq,
				avgfreq;

	/*
	 * Obtain number of distinct data values in raw relation.
	 */
	ndistinct = stats.stadistinct;
	if (ndistinct < 0.0f)
		ndistinct = -ndistinct * baserel.tuples;

	if (ndistinct <= 0.0f)		/* ensure we can divide */
	{
		return 0.1f;
	}

	/* Also compute avg freq of all distinct data values in raw relation */
	avgfreq = (1.0f - stats.stanullfrac) / ndistinct;

	/*
	 * Adjust ndistinct to account for restriction clauses.  Observe we
	 * are assuming that the data distribution is affected uniformly by
	 * the restriction clauses!
	 *
	 * XXX Possibly better way, but much more expensive: multiply by
	 * selectivity of rel's restriction clauses that mention the target
	 * Var.
	 */
	ndistinct *= baserel.rows / baserel.tuples;

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
	mcvfreq = stats.mcvfreq;

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
	return tuples * (MAXALIGN(width) + MAXALIGN(SIZE_OF_HEAP_TUPLE_DATA));
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
static struct Cost
cost_baserel(BaseRelation<BitmapsetN> &base_rel){
    return base_rel.cost;
}


#endif
