/*------------------------------------------------------------------------
 *
 * gpuqo_remapper.cuh
 *      class for remapping relations to other indices
 *
 * src/backend/optimizer/gpuqo/gpuqo_remapper.cuh
 *
 *-------------------------------------------------------------------------
 */

#ifndef GPUQO_REMAPPER_CUH
#define GPUQO_REMAPPER_CUH

#include <list>
#include "optimizer/gpuqo_common.h"
#include "gpuqo_planner_info.cuh"

using namespace std;

template<typename BitmapsetN>
struct remapper_transf_el_t {
    BitmapsetN from_relid;
    int to_idx;
    QueryTree<BitmapsetN> *qt;
};

template<typename BitmapsetN>
class Remapper{
private:
    list<remapper_transf_el_t<BitmapsetN> > transf;

    void countEqClasses(GpuqoPlannerInfo<BitmapsetN>* info, 
                                            int* n, int* n_sels, int *n_fk, int *n_stats);
    BitmapsetN remapRelid(BitmapsetN id);
    BitmapsetN remapRelidNoComposite(BitmapsetN id);
    BitmapsetN remapRelidInv(BitmapsetN id);
    void remapEdgeTable(BitmapsetN* edge_table_from, BitmapsetN* edge_table_to,
                        bool ignore_composite=false);
    void remapBaseRels(BaseRelation<BitmapsetN>* base_rels_from,
                        BaseRelation<BitmapsetN>* base_rels_to);
    void remapEqClass(BitmapsetN* eq_class_from, float* sels_from, 
                    BitmapsetN* fks_from, VarStat* stats_from,
                    GpuqoPlannerInfo<BitmapsetN>* info_from,
                    int off_sels_from, int off_fks_from, 
                    BitmapsetN* eq_class_to, float* sels_to, BitmapsetN* fks_to,
                    VarStat* stats_to);

public:
    Remapper<BitmapsetN>(list<remapper_transf_el_t<BitmapsetN>> _transf);

    GpuqoPlannerInfo<BitmapsetN>* remapPlannerInfo(
                                            GpuqoPlannerInfo<BitmapsetN>* info);
    void remapQueryTree(QueryTree<BitmapsetN>* qt);
};

#endif              // GPUQO_REMAPPER_CUH
