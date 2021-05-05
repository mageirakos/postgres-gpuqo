/*------------------------------------------------------------------------
 *
 * gpuqo_remapper.cu
 *      implementation for class for remapping relations to other indices
 *
 * src/backend/optimizer/gpuqo/gpuqo_remapper.cu
 *
 *-------------------------------------------------------------------------
 */

#include "gpuqo_remapper.cuh"
#include "gpuqo_cost.cuh"

template<typename BitmapsetN>
Remapper<BitmapsetN>::Remapper(list<remapper_transf_el_t<BitmapsetN> > _transf) 
                                : transf(_transf) {}

template<typename BitmapsetN>
void Remapper<BitmapsetN>::countEqClasses(GpuqoPlannerInfo<BitmapsetN>* info, 
                                        int* n, int* n_sels)
{
    *n = 0;
    *n_sels = 0;

    for (int i = 0; i < info->n_eq_classes; i++){
        bool found = false;
        for (remapper_transf_el_t<BitmapsetN> &e : transf){
            if (info->eq_classes[i].isSubset(e.from_relid)){
                found = true;
                break;
            }
        }
        if (!found){
            (*n)++;
            (*n_sels) += eqClassNSels(info->eq_classes[i].size());
        }
    }
}

template<typename BitmapsetN>
BitmapsetN Remapper<BitmapsetN>::remapRelid(BitmapsetN id)
{
    BitmapsetN out = BitmapsetN(0);
    for (remapper_transf_el_t<BitmapsetN> &e : transf){
        if (e.from_relid.intersects(id)){
            out.set(e.to_idx+1);
        }
    }

    return out;
}

template<typename BitmapsetN>
BitmapsetN Remapper<BitmapsetN>::remapRelidInv(BitmapsetN id)
{
    BitmapsetN out = BitmapsetN(0);
    for (remapper_transf_el_t<BitmapsetN> &e : transf){
        if (id.isSet(e.to_idx+1)){
            out |= e.from_relid;
        }
    }

    return out;
}

template<typename BitmapsetN>
void Remapper<BitmapsetN>::remapEdgeTable(BitmapsetN* edge_table_from, 
                                            BitmapsetN* edge_table_to)
{
    for (remapper_transf_el_t<BitmapsetN> &e : transf){
        edge_table_to[e.to_idx] = BitmapsetN(0);
        
        BitmapsetN temp = e.from_relid;
        while(!temp.empty()){
            int from_idx = temp.lowestPos()-1;

            edge_table_to[e.to_idx] |= remapRelid(edge_table_from[from_idx]);

            temp.unset(from_idx+1);
        }
    }
}

template<typename BitmapsetN>
void Remapper<BitmapsetN>::remapBaseRels(
                                BaseRelation<BitmapsetN>* base_rels_from, 
                                BaseRelation<BitmapsetN>* base_rels_to)
{

    for (remapper_transf_el_t<BitmapsetN> &e : transf){
        if (e.qt != NULL){
            base_rels_to[e.to_idx].id = remapRelid(e.from_relid);
            base_rels_to[e.to_idx].rows = e.qt->rows;
            base_rels_to[e.to_idx].cost = e.qt->cost;
        } else {
            base_rels_to[e.to_idx] = base_rels_from[e.from_relid.lowestPos()-1];
            base_rels_to[e.to_idx].id = remapRelid(e.from_relid);
        }
    }
}

template<typename BitmapsetN>
void Remapper<BitmapsetN>::remapEqClass(BitmapsetN* eq_class_from,
                                        float* sels_from,
                                        BitmapsetN* eq_class_to,
                                        float* sels_to,
                                        GpuqoPlannerInfo<BitmapsetN>* from_info)
{
    *eq_class_to = remapRelid(*eq_class_from);

    int s_from = eq_class_from->size();
    int n_from = eqClassNSels(s_from);
    int s_to = eq_class_to->size();
    int n_to = eqClassNSels(s_to);

    for (int idx_l_to = 0; idx_l_to < s_to; idx_l_to++){
        for (int idx_r_to = idx_l_to+1; idx_r_to < s_to; idx_r_to++){
            BitmapsetN id_l_to = expandToMask(BitmapsetN::nth(idx_l_to), 
                                                *eq_class_to); 
            BitmapsetN id_l_from = remapRelidInv(id_l_to);

            BitmapsetN id_r_to = expandToMask(BitmapsetN::nth(idx_r_to), 
                                                *eq_class_to); 
            BitmapsetN id_r_from = remapRelidInv(id_r_to);

            int sels_to_idx = eqClassIndex(idx_l_to, idx_r_to, s_to);

            if (id_l_from.size() > 1 || id_r_from.size()){
                sels_to[sels_to_idx] = estimate_join_selectivity(id_l_from, 
                                                        id_r_from, from_info);
            } else {
                int idx_l_from = (id_l_from.allLower() & *eq_class_from).size();
                int idx_r_from = (id_r_from.allLower() & *eq_class_from).size();

                int sels_from_idx = eqClassIndex(idx_l_from, idx_r_from, 
                                                    s_from);
    
                sels_to[sels_to_idx] = sels_from[sels_from_idx];
            }
        }
    }
}

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN> *Remapper<BitmapsetN>::remapPlannerInfo(
                                        GpuqoPlannerInfo<BitmapsetN>* old_info)
{
    int n_rels = transf.size();
    int n_fk_selecs = 0; // TODO: not supported atm
    int n_eq_classes, n_eq_class_sels; 
    countEqClasses(old_info, &n_eq_classes, &n_eq_class_sels); 

    unsigned int size = sizeof(GpuqoPlannerInfo<BitmapsetN>);
	size += sizeof(unsigned int) * n_fk_selecs;
	size += sizeof(float) * n_fk_selecs;
	size += sizeof(BitmapsetN) * n_eq_classes;
	size += sizeof(float) * n_eq_class_sels;
	size += ceil_div(size, 8)*8; // ceil to 64 bits multiples

	char* p = new char[size];

	GpuqoPlannerInfo<BitmapsetN> *info = (GpuqoPlannerInfo<BitmapsetN>*) p;
	p += sizeof(GpuqoPlannerInfo<BitmapsetN>);

	info->size = size;
	info->n_rels = n_rels;
	info->n_iters = old_info->n_iters;

    remapEdgeTable(old_info->edge_table, info->edge_table);
    remapEdgeTable(old_info->indexed_edge_table, info->indexed_edge_table);

    if (gpuqo_spanning_tree_enable)
        remapEdgeTable(old_info->subtrees, info->subtrees);

	info->n_fk_selecs = n_fk_selecs;

    // TODO fksels
	info->fk_selec_idxs = (unsigned int*) p;
	p += sizeof(unsigned int) * n_fk_selecs;
	info->fk_selec_sels = (float*) p;
	p += sizeof(float) * n_fk_selecs;

	remapBaseRels(old_info->base_rels, info->base_rels);

	info->n_eq_classes = n_eq_classes;
	info->n_eq_class_sels = n_eq_class_sels;

	info->eq_classes = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_classes;
	info->eq_class_sels = (float*) p;
	p += sizeof(float) * info->n_eq_class_sels;

    int offset = 0, old_offset = 0, j = 0;
	for (int i = 0; i < old_info->n_eq_classes; i++){
        bool found = false;
        for (remapper_transf_el_t<BitmapsetN> &e : transf){
            if (old_info->eq_classes[i].isSubset(e.from_relid)){
                found = true;
                break;
            }
        }
        if (!found){
            remapEqClass(
                &old_info->eq_classes[i], &old_info->eq_class_sels[old_offset], 
                &info->eq_classes[j], &info->eq_class_sels[offset], 
                old_info
            );
            offset += eqClassNSels(info->eq_classes[j].size());
            j++;
        }

        old_offset += eqClassNSels(old_info->eq_classes[i].size());
    }

	return info;
}

template<typename BitmapsetN>
void Remapper<BitmapsetN>::remapQueryTree(QueryTree<BitmapsetN>* qt){
    if (qt->id.size() == 1){
        int idx = qt->id.lowestPos() - 1;

        for (remapper_transf_el_t<BitmapsetN> &e : transf){
            if (e.qt != NULL && e.to_idx == idx){
                *qt = *e.qt;
                
                // TODO check
                delete e.qt;

                return;
            }
        }
        // otherwise
        qt->id = remapRelidInv(qt->id);
    } else {       
        qt->id = remapRelidInv(qt->id);

        remapQueryTree(qt->left);
        remapQueryTree(qt->right);
    }
}

template class Remapper<Bitmapset32>;
template class Remapper<Bitmapset64>;
