/*------------------------------------------------------------------------
 *
 * gpuqo_main.c
 *	  solution to the query optimization problem
 *	  using GPU acceleration
 *
 * src/backend/optimizer/gpuqo/gpuqo_main.c
 *
 *-------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "postgres.h"
#include "miscadmin.h"

#include "utils/selfuncs.h"
#include "utils/lsyscache.h"

#include "access/htup_details.h"

#include "nodes/nodeFuncs.h"

#include "catalog/pg_statistic.h"

#include "optimizer/optimizer.h"
#include "optimizer/cost.h"
#include "optimizer/gpuqo.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"

#define GPUQO_INFO

#ifdef OPTIMIZER_DEBUG
#ifndef GPUQO_INFO
#define GPUQO_INFO
#endif
#endif

int gpuqo_algorithm;

static BaseRelationC makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
static EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels);
static void printQueryTree(QueryTreeC* qt, int indent);
static void printEdges(GpuqoPlannerInfoC* info);
static RelOptInfo* queryTree2Plan(QueryTreeC* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels);
static void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfoC* info, int n_rels);
static void set_eq_class_foreign_keys(PlannerInfo *root, GpuqoPlannerInfoC *info,
                            Relids outer_relids, Relids inner_relids,
                            SpecialJoinInfo *sjinfo, List **restrictlist);

/*
 * gpuqo_check_can_run
 *	  checks that there are no join order contraints.
 */
bool 
gpuqo_check_can_run(PlannerInfo* root)
{
    if (list_length(root->join_info_list) != 0)
        return false;

    if (list_length(root->placeholder_list) != 0)
        return false;

    return true;
}

void printQueryTree(QueryTreeC* qt, int indent){
    int i;

    /* since this function recurses, it could be driven to stack overflow */
	check_stack_depth();

    if (qt == NULL)
        return;

    for (i = 0; i<indent; i++)
        printf(" ");
    printf("%lu (rows=%.0f, cost=%.2f..%.2f, width=%d)\n", qt->id->words[0], qt->rows, qt->cost.startup, qt->cost.total, qt->width);

    printQueryTree(qt->left, indent + 2);
    printQueryTree(qt->right, indent + 2);
}

void printEdges(GpuqoPlannerInfoC* info){
    printf("\nEdges:\n");
    for (int i = 0; i < info->n_rels; i++){
        RelationID edges = info->edge_table[i];
        int idx = -1;
        printf("%d:", i+1);
        while ((idx = bms_next_member(edges, idx)) >= 0){
            EqClassInfo *ec = info->eq_classes;
            
            printf(" %d ", idx);

            if (ec != NULL){
                printf("(");
                while (ec != NULL){
                    if (bms_is_member(i+1, ec->relids) && bms_is_member(idx, ec->relids)){
                        int size = bms_num_members(ec->relids);
                        int idx_l = bms_member_index(ec->relids, i+1);
                        int idx_r = bms_member_index(ec->relids, idx);
                        int idx_sel = eqClassIndex(idx_l, idx_r, size);

                        printf("%lu: %e, ", ec->relids->words[0], ec->sels[idx_sel]);
                    }
                    ec = ec->next;
                }
                printf(");");
            }
        }
        printf("\n");
    }
}

void
set_eq_class_foreign_keys(PlannerInfo *root,
                            GpuqoPlannerInfoC *info,
                            Relids outer_relids,
                            Relids inner_relids,
                            SpecialJoinInfo *sjinfo,
                            List **restrictlist)
{
	JoinType	jointype = sjinfo->jointype;
	List	   *worklist = *restrictlist;
	ListCell   *lc;

	/* Consider each FK constraint that is known to match the query */
	foreach(lc, root->fkey_list)
	{
		ForeignKeyOptInfo *fkinfo = (ForeignKeyOptInfo *) lfirst(lc);
		bool		ref_is_outer;
		ListCell   *cell;
		ListCell   *next;

		/*
		 * This FK is not relevant unless it connects a baserel on one side of
		 * this join to a baserel on the other side.
		 */
		if (bms_is_member(fkinfo->con_relid, outer_relids) &&
			bms_is_member(fkinfo->ref_relid, inner_relids))
			ref_is_outer = false;
		else if (bms_is_member(fkinfo->ref_relid, outer_relids) &&
				 bms_is_member(fkinfo->con_relid, inner_relids))
			ref_is_outer = true;
		else
			continue;

		if ((jointype == JOIN_SEMI || jointype == JOIN_ANTI) &&
			(ref_is_outer || bms_membership(inner_relids) != BMS_SINGLETON))
			continue;

		if (worklist == *restrictlist)
			worklist = list_copy(worklist);

		for (cell = list_head(worklist); cell; cell = next)
		{
			RestrictInfo *rinfo = (RestrictInfo *) lfirst(cell);
			int			i;
			bool		matches = false;

			next = lnext(cell);

			for (i = 0; i < fkinfo->nkeys; i++)
			{
				if (rinfo->parent_ec)
				{
					if (fkinfo->eclass[i] == rinfo->parent_ec)
					{
						matches = true;
						break;
					}
				}
			}
			if (matches)
			{
                EqClassInfo* eq = info->eq_classes;

                while (eq != NULL){
                    if (eq->eclass == rinfo->parent_ec)
                    {
                        Bitmapset **bms = &eq->fk[
                                bms_member_index(eq->relids, fkinfo->con_relid)
                        ];
                        *bms = bms_add_member(*bms, fkinfo->ref_relid);
                        
                        break;
                    }
                    eq = eq->next;
                }
            }
		}
    }
}

RelOptInfo* queryTree2Plan(QueryTreeC* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels){
    RelOptInfo* left_rel = NULL;
    RelOptInfo* right_rel = NULL;
    RelOptInfo* this_rel = NULL;
    ListCell* lc;

    /* since this function recurses, it could be driven to stack overflow */
	check_stack_depth();

    // baserel
    if (qt->left == 0 || qt->right == 0){
        foreach(lc, initial_rels){
            RelOptInfo* base_rel = (RelOptInfo*) lfirst(lc);
            if (bms_equal(base_rel->relids, qt->id)){
                this_rel = base_rel;
                break;
            }
        }
    } else { // joinrel
        left_rel = queryTree2Plan(qt->left, level-1, root, n_rels, initial_rels);
        right_rel = queryTree2Plan(qt->right, level-1, root, n_rels,initial_rels);
        this_rel = make_join_rel(root, left_rel, right_rel);
        if (this_rel){
            /* Create paths for partitionwise joins. */
            generate_partitionwise_join_paths(root, this_rel);

            /*
                * Except for the topmost scan/join rel, consider gathering
                * partial paths.  We'll do the same for the topmost scan/join
                * rel once we know the final targetlist (see
                * grouping_planner).
                */
            if (level != n_rels)
                generate_gather_paths(root, this_rel, false);

            /* Find and save the cheapest paths for this joinrel */
            set_cheapest(this_rel);
        } 
    }

    if (this_rel == NULL) {
        printf("WARNING: Found NULL RelOptInfo*\n");
    }

    // Assert(this_rel->rows == qt->rows);

    // clean-up the query tree
    pfree(qt);

    return this_rel;
}

BaseRelationC makeBaseRelation(RelOptInfo* rel, PlannerInfo* root){
    BaseRelationC baserel;
    
    baserel.rows = rel->rows;
    baserel.tuples = rel->tuples;
    baserel.width = rel->reltarget->width;
    baserel.pages = rel->pages;
    baserel.cost.total = rel->cheapest_total_path->total_cost;
    baserel.cost.startup = rel->cheapest_total_path->startup_cost;
    baserel.id = bms_copy(rel->relids);

    return baserel;
}

EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels){
    ListCell* lc;

    EdgeMask* edge_table = palloc0(sizeof(EdgeMask)*n_rels);
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);

        if (list_length(ec->ec_members) <= 1)
			continue;

        for (int i=1; i <= n_rels; i++){
            if (bms_is_member(i, ec->ec_relids)){
                edge_table[i-1] = bms_add_members(edge_table[i-1], ec->ec_relids);
                edge_table[i-1] = bms_del_member(edge_table[i-1], i);
            }
        }
    }

    return edge_table;
}

static VarStatC extractStats(PlannerInfo *root, Node *hashkey)
{
    VarStatC     out;
	VariableStatData vardata;
	bool		isdefault;
	AttStatsSlot sslot;

    out.mcvfreq = 0;
    out.stadistinct = 0;
    out.stanullfrac = 0;

	examine_variable(root, hashkey, 0, &vardata);

	/* Look up the frequency of the most common value, if available */
	out.mcvfreq = 0.0;

	if (HeapTupleIsValid(vardata.statsTuple))
	{
		if (get_attstatsslot(&sslot, vardata.statsTuple,
							 STATISTIC_KIND_MCV, InvalidOid,
							 ATTSTATSSLOT_NUMBERS))
		{
			/*
			 * The first MCV stat is for the most common value.
			 */
			if (sslot.nnumbers > 0)
				out.mcvfreq = (float) sslot.numbers[0];
			free_attstatsslot(&sslot);
		}
	}

	/* Get number of distinct values */
	out.stadistinct = get_variable_numdistinct(&vardata, &isdefault);

	/*
	 * If ndistinct isn't real, punt.  We normally return 0.1, but if the
	 * mcv_freq is known to be even higher than that, use it instead.
	 */
	if (isdefault)
	{
		ReleaseVariableStats(vardata);
		return out;
	}

	/* Get fraction that are null */
	if (HeapTupleIsValid(vardata.statsTuple))
	{
		Form_pg_statistic stats;

		stats = (Form_pg_statistic) GETSTRUCT(vardata.statsTuple);
		out.stanullfrac = stats->stanullfrac;
	}
	else
		out.stanullfrac = 0.0;

	ReleaseVariableStats(vardata);

    return out;
}

void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfoC* info, int n_rels){
    ListCell* lc_inner;
    ListCell* lc_outer;
    ListCell* lc_inner_path;
    ListCell* lc_restrictinfo;
    int i_out, i_in;

    info->indexed_edge_table = (EdgeMask*) palloc0(n_rels * sizeof(EdgeMask));

    i_out = 0;
    foreach(lc_outer, initial_rels){
        RelOptInfo* rel_outer = (RelOptInfo*) lfirst(lc_outer);
        i_in = 0;
        foreach(lc_inner, initial_rels){
            RelOptInfo* rel_inner = (RelOptInfo*) lfirst(lc_inner);
            
            if (bms_overlap(info->edge_table[i_out], rel_inner->relids)){
                List *restrictlist = NULL;
                SpecialJoinInfo sjinfo;
                RelOptInfo *join_rel;
                
                sjinfo.type = T_SpecialJoinInfo;
                sjinfo.jointype = JOIN_INNER;
                /* we don't bother trying to make the remaining fields valid */
                sjinfo.lhs_strict = false;
                sjinfo.delay_upper_joins = false;
                sjinfo.semi_can_btree = false;
                sjinfo.semi_can_hash = false;
                sjinfo.semi_operators = NIL;
                sjinfo.semi_rhs_exprs = NIL;
                sjinfo.min_lefthand = rel_outer->relids;
                sjinfo.min_righthand = rel_inner->relids;
                sjinfo.syn_lefthand = rel_outer->relids;
                sjinfo.syn_righthand = rel_inner->relids;

                join_rel = build_join_rel(root,
                    bms_union(rel_outer->relids, 
                                rel_inner->relids),
                    rel_outer,
                    rel_inner,
                    &sjinfo,
                    NULL
                ); 
                
                // I just care about restrictlist
                // but I need to recompute it since it may have been cleaned
                // out by fk selectivities
                restrictlist = build_joinrel_restrictlist(root, join_rel,
                    rel_outer, rel_inner
                );
                
                // recompute selectivities (they will be cached inside the 
                // RestrictInfo), if necessary
                clauselist_selectivity(root,
										restrictlist,
										0,
										sjinfo.jointype,
										&sjinfo);

                foreach(lc_restrictinfo, restrictlist){
                    int size, idx_l, idx_r, idx;
                    RestrictInfo *rinfo= (RestrictInfo*) lfirst(lc_restrictinfo);
                    EqClassInfo* ec = info->eq_classes;

                    if (rinfo->norm_selec == -1)
                        continue;


                    while (ec != NULL){
                        if (bms_equal(ec->relids, rinfo->parent_ec->ec_relids))
                            break;
                        ec = ec->next;

                    };

                    if (ec == NULL){
                        size_t n_sels;
                        
                        ec = (EqClassInfo*) palloc(sizeof(EqClassInfo));
                        ec->next = info->eq_classes;
                        info->eq_classes = ec;
                        ec->relids = bms_copy(rinfo->parent_ec->ec_relids);
                        size = bms_num_members(ec->relids);
                        n_sels = eqClassNSels(size);
                        ec->sels = (float*) palloc(sizeof(float)*n_sels);
                        ec->eclass = rinfo->parent_ec;
                        ec->fk = (RelationID*) palloc(sizeof(RelationID)*size);
                        ec->stats = (VarStatC*) palloc(sizeof(VarStatC)*size);
                        for (int i_fk = 0; i_fk < size; i_fk++){
                            ec->fk[i_fk] = NULL;
                            ec->stats[i_fk].mcvfreq = 0;
                            ec->stats[i_fk].stadistinct = 0;
                            ec->stats[i_fk].stanullfrac = 0;
                        }
                        info->n_eq_classes++;
                        info->n_eq_class_sels += n_sels;
                        info->n_eq_class_fks += size;
                        info->n_eq_class_stats += size;
                    } else {
                        size = bms_num_members(ec->relids);
                    }

                    idx_l = bms_member_index(ec->relids, i_out+1);
                    idx_r = bms_member_index(ec->relids, i_in+1);
                    if (idx_l < idx_r){ // prevent duplicates
                        idx = eqClassIndex(idx_l, idx_r, size);
                        ec->sels[idx] = rinfo->norm_selec;

                        if (bms_is_subset(rinfo->right_relids,
							  rel_inner->relids)) {

                            ec->stats[idx_r] = extractStats(root,
                                                get_rightop(rinfo->clause));

                        } else {
                            ec->stats[idx_r] = extractStats(root,
                                                get_leftop(rinfo->clause));
                        }
                        
                        if (bms_is_subset(rinfo->right_relids,
							  rel_outer->relids)) {

                            ec->stats[idx_l] = extractStats(root,
                                                get_rightop(rinfo->clause));

                        } else {
                            ec->stats[idx_l] = extractStats(root,
                                                get_leftop(rinfo->clause));
                        }

                    }
                    
                        
                }

                foreach(lc_inner_path, rel_inner->pathlist){
                    Path* path = (Path*) lfirst(lc_inner_path);
                    if (path->pathtype == T_IndexScan){
                        if (bms_num_members(PATH_REQ_OUTER(path)) == 1 
                                && bms_overlap(PATH_REQ_OUTER(path),
                                    rel_outer->relids
                                )){
                            info->indexed_edge_table[i_in] = bms_add_member(info->indexed_edge_table[i_in], i_out+1);
                            break;
                        }
                    }
                }

                set_eq_class_foreign_keys(root, info,
                                            rel_outer->relids,
                                            rel_inner->relids,
                                            &sjinfo,
                                            &restrictlist);
            }
            i_in++;
        }
        i_out++;
    }
}

static void
EqClassInfo_list__free(EqClassInfo* node) 
{
    EqClassInfo* prev;

    while (node) {
        prev = node;
        node = node->next;
        pfree(prev);
    }
}

static void
GpuqoPlannerInfoC__free(GpuqoPlannerInfoC* info) 
{
    EqClassInfo_list__free(info->eq_classes);
    pfree(info->indexed_edge_table);
    pfree(info->base_rels);
    pfree(info->edge_table);
    pfree(info);
}

/*
 * gpuqo
 *	  solution of the query optimization problem
 *	  using GPU acceleration (CUDA)
 */

RelOptInfo *
gpuqo(PlannerInfo *root, int n_rels, List *initial_rels)
{
    ListCell* lc;
    int i;
    RelOptInfo* rel;
    GpuqoPlannerInfoC* info;
    QueryTreeC* query_tree;

#ifdef OPTIMIZER_DEBUG
    printf("Hello, here is gpuqo!\n");
#endif

    info = (GpuqoPlannerInfoC*) palloc(sizeof(GpuqoPlannerInfoC));
    
    info->base_rels = (BaseRelationC*) palloc(n_rels * sizeof(BaseRelationC));

    info->n_rels = n_rels;
    info->eq_classes = NULL;
    info->n_eq_classes = 0;
    info->n_eq_class_sels = 0;
    info->n_eq_class_fks = 0;
    info->n_eq_class_stats = 0;

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        info->base_rels[i++] = makeBaseRelation(rel, root);
    }
    info->edge_table = makeEdgeTable(root, n_rels);
    fillSelectivityInformation(root, initial_rels, info, n_rels);

#ifdef GPUQO_INFO
    printEdges(info);
#endif

    query_tree = gpuqo_run(gpuqo_algorithm, info);
    
    GpuqoPlannerInfoC__free(info);
    
#ifdef GPUQO_INFO
    printQueryTree(query_tree, 2);
    printf("gpuqo cost is %f\n", query_tree->cost.total);
#endif

	return queryTree2Plan(query_tree, n_rels, root, n_rels, initial_rels);
}
