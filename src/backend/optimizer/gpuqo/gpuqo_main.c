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

#include "postgres.h"
#include "miscadmin.h"

#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/gpuqo.h"

int gpuqo_algorithm;

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root);
EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels);
Bitmapset64 convertBitmapset(Bitmapset* set);
void printQueryTree(QueryTree* qt, int indent);
void printEdges(GpuqoPlannerInfo* info);
RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels);
void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info, int n_rels);

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

void printQueryTree(QueryTree* qt, int indent){
    int i;

    /* since this function recurses, it could be driven to stack overflow */
	check_stack_depth();

    if (qt == NULL)
        return;

    for (i = 0; i<indent; i++)
        printf(" ");
    printf("%llu (rows=%.0f, cost=%.2f)\n", qt->id, qt->rows, qt->cost);

    printQueryTree(qt->left, indent + 2);
    printQueryTree(qt->right, indent + 2);
}

void printEdges(GpuqoPlannerInfo* info){
    for (int i = 0; i < info->n_rels; i++){
        RelationID edges = info->edge_table[i];
        printf("%d:", i+1);
        while (edges != BMS64_EMPTY){
            int idx = BMS64_LOWEST_POS(edges)-1;
            EqClassInfo *ec = info->eq_classes;
            
            printf(" %d", idx);

            if (ec != NULL){
                printf("(");
                while (ec != NULL){
                    if (BMS64_IS_SET(ec->relids, i+1) && BMS64_IS_SET(ec->relids, idx)){
                        int size = BMS64_SIZE(ec->relids);
                        int idx_l = BMS64_SIZE(
                            BMS64_INTERSECTION(
                                BMS64_SET_ALL_LOWER(BMS64_NTH(i+1)),
                                ec->relids
                            )
                        );
                        int idx_r = BMS64_SIZE(
                            BMS64_INTERSECTION(
                                BMS64_SET_ALL_LOWER(BMS64_NTH(idx)),
                                ec->relids
                            )
                        );
                        printf("%llu: %f, ", ec->relids, ec->sels[idx_l*size+idx_r]);
                    }
                    ec = ec->next;
                }
                printf(") ");
            }

            edges = BMS64_UNSET(edges, idx);
        }
        printf("\n");
    }
}

RelOptInfo* queryTree2Plan(QueryTree* qt, int level, PlannerInfo *root, int n_rels, List *initial_rels){
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
            if (base_rel->relids->words[0] & qt->id){
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
        printf("WARNING: Found NULL RelOptInfo*: %llu (%llu, %llu)\n", 
                qt->id, 
                qt->left ? qt->left->id : 0, 
                qt->right ? qt->right->id : 0
        );
    }

    // clean-up the query tree
    free(qt);

    return this_rel;
}

Bitmapset64 convertBitmapset(Bitmapset* set){
    if (set->nwords > 1){
        printf("WARNING: only relids of 64 bits are supported!\n");
    }
    return set->words[0];
}

BaseRelation makeBaseRelation(RelOptInfo* rel, PlannerInfo* root){
    BaseRelation baserel;
    
    baserel.rows = rel->rows;
    baserel.tuples = rel->tuples;
    baserel.id = convertBitmapset(rel->relids);

    return baserel;
}

EdgeMask* makeEdgeTable(PlannerInfo* root, int n_rels){
    ListCell* lc;

    EdgeMask* edge_table = gpuqo_malloc(sizeof(EdgeMask)*n_rels);
    memset(edge_table, 0, sizeof(EdgeMask)*n_rels);
    foreach(lc, root->eq_classes){
        EquivalenceClass *ec = (EquivalenceClass *) lfirst(lc);
        EdgeMask ec_relids = convertBitmapset(ec->ec_relids);

        if (list_length(ec->ec_members) <= 1)
			continue;

        for (int i=1; i <= n_rels; i++){
            RelationID baserelid = BMS64_NTH(i);
            if (BMS64_INTERSECTS(ec_relids, baserelid)){
                edge_table[i-1] = BMS64_UNION(
                        edge_table[i-1],
                        BMS64_DIFFERENCE(ec_relids, baserelid)
                );
            }
            
        }
    }

    return edge_table;
}

void fillSelectivityInformation(PlannerInfo *root, List *initial_rels, GpuqoPlannerInfo* info, int n_rels){
    ListCell* lc_inner;
    ListCell* lc_outer;
    ListCell* lc_inner_path;
    ListCell* lc_restrictinfo;
    List	   *restrictlist = NULL;
    int i, j;

    info->indexed_edge_table = (EdgeMask*) gpuqo_malloc(n_rels * sizeof(EdgeMask));
    memset(info->indexed_edge_table, 0, n_rels * sizeof(EdgeMask));

    i = 0;
    foreach(lc_outer, initial_rels){
        RelOptInfo* rel_outer = (RelOptInfo*) lfirst(lc_outer);
        j = 0;
        foreach(lc_inner, initial_rels){
            RelOptInfo* rel_inner = (RelOptInfo*) lfirst(lc_inner);
            
            if (BMS64_INTERSECTS(info->edge_table[i], convertBitmapset(rel_inner->relids))){
                List *restrictlist = NULL;
                SpecialJoinInfo sjinfo;
                
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

                build_join_rel(root,
                    bms_union(rel_outer->relids, 
                                rel_inner->relids),
                    rel_outer,
                    rel_inner,
                    &sjinfo,
                    &restrictlist
                ); // I just care about restrictlist

                foreach(lc_restrictinfo, restrictlist){
                    int size, idx_l, idx_r;
                    RestrictInfo *rinfo= (RestrictInfo*) lfirst(lc_restrictinfo);
                    EqClassInfo* ec = info->eq_classes;

                    if (rinfo->norm_selec == -1)
                        continue;


                    while (ec != NULL){
                        if (ec->relids == convertBitmapset(rinfo->parent_ec->ec_relids))
                            break;
                        ec = ec->next;

                    };

                    if (ec == NULL){
                        ec = (EqClassInfo*) gpuqo_malloc(sizeof(EqClassInfo));
                        ec->next = info->eq_classes;
                        info->eq_classes = ec;
                        ec->relids = convertBitmapset(rinfo->parent_ec->ec_relids);
                        size = BMS64_SIZE(ec->relids);
                        ec->sels = (double*) gpuqo_malloc(sizeof(double)*size*size);
                    } else {
                        size = BMS64_SIZE(ec->relids);
                    }

                    idx_l = BMS64_SIZE(
                        BMS64_INTERSECTION(
                            BMS64_SET_ALL_LOWER(BMS64_NTH(i+1)),
                            ec->relids
                        )
                    );
                    idx_r = BMS64_SIZE(
                        BMS64_INTERSECTION(
                            BMS64_SET_ALL_LOWER(BMS64_NTH(j+1)),
                            ec->relids
                        )
                    );
                    ec->sels[idx_l*size+idx_r] = rinfo->norm_selec;
                    ec->sels[idx_r*size+idx_l] = rinfo->norm_selec;
                }

                foreach(lc_inner_path, rel_inner->pathlist){
                    Path* path = (Path*) lfirst(lc_inner_path);
                    if (path->pathtype == T_IndexScan){
                        if (bms_num_members(PATH_REQ_OUTER(path)) == 1 
                                && bms_overlap(PATH_REQ_OUTER(path),
                                    rel_outer->relids
                                )){
                            info->indexed_edge_table[j] = BMS64_SET(info->indexed_edge_table[j], i+1);
                            break;
                        }
                    }
                }
                
            }
            j++;
        }
        i++;
    }
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
    GpuqoPlannerInfo* info;
    QueryTree* query_tree;

#ifdef OPTIMIZER_DEBUG
    printf("Hello, here is gpuqo!\n");
#endif

    info = (GpuqoPlannerInfo*) gpuqo_malloc(sizeof(GpuqoPlannerInfo));
    info->base_rels = (BaseRelation*) gpuqo_malloc(n_rels * sizeof(BaseRelation));
    info->n_rels = n_rels;
    info->eq_classes = NULL;

    i = 0;
    foreach(lc, initial_rels){
        rel = (RelOptInfo *) lfirst(lc);
        info->base_rels[i++] = makeBaseRelation(rel, root);
    }
    info->edge_table = makeEdgeTable(root, n_rels);
    fillSelectivityInformation(root, initial_rels, info, n_rels);

#ifdef OPTIMIZER_DEBUG
    printEdges(info);
#endif

    switch (gpuqo_algorithm)
    {
    case GPUQO_DPSIZE:
        query_tree = gpuqo_dpsize(info);
        break;
    case GPUQO_DPSUB:
        query_tree = gpuqo_dpsub(info);
        break;
    case GPUQO_CPU_DPSIZE:
        query_tree = gpuqo_cpu_dpsize(info);
        break;
    case GPUQO_CPU_DPSUB:
        query_tree = gpuqo_cpu_dpsub(info);
        break;
    case GPUQO_CPU_DPCCP:
        query_tree = gpuqo_cpu_dpccp(info);
        break;
    case GPUQO_DPE_DPSIZE:
        query_tree = gpuqo_dpe_dpsize(info);
        break;
    case GPUQO_DPE_DPSUB:
        query_tree = gpuqo_dpe_dpsub(info);
        break;
    case GPUQO_DPE_DPCCP:
        query_tree = gpuqo_dpe_dpccp(info);
        break;
    default: 
        // impossible branch but without it the compiler complains
        query_tree = NULL;
        break;
    }
    
    gpuqo_free(info->base_rels);
    gpuqo_free(info->edge_table);
    gpuqo_free(info);
    
#ifdef OPTIMIZER_DEBUG
    printQueryTree(query_tree, 2);
#endif

	return queryTree2Plan(query_tree, n_rels, root, n_rels, initial_rels);
}
