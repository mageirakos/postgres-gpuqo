/*-------------------------------------------------------------------------
 *
 * gpuqo_planner_info.cu
 *	  structure for conversion from C to C++ optimized structure.
 *
 * src/include/optimizer/gpuqo_planner_info.cu
 *
 *-------------------------------------------------------------------------
 */

#include <iostream>

#include <optimizer/gpuqo_common.h>
#include "gpuqo_planner_info.cuh"
#include "gpuqo_cost.cuh"

extern "C" void *palloc0(size_t size);
extern "C" void *palloc(size_t size);

static size_t bitmapset_size(int nwords){
	return (offsetof(gpuqo_c::Bitmapset, words) + (nwords) * sizeof(gpuqo_c::bitmapword));
}

template<typename Bitmapset>
Bitmapset convertBitmapset(gpuqo_c::Bitmapset* set);

template<>
Bitmapset32 convertBitmapset<Bitmapset32>(gpuqo_c::Bitmapset* set){
	if (set == NULL)
		return Bitmapset32(0);


    if (set->nwords > 1){
        printf("WARNING: only relids of 32 bits are supported!\n");
    }
    if (set->words[0] & 0xFFFFFFFF00000000ULL){
        printf("WARNING: only relids of 32 bits are supported!\n");
    }
    return Bitmapset32(set->words[0] & 0xFFFFFFFFULL);
}
template<>
Bitmapset64 convertBitmapset<Bitmapset64>(gpuqo_c::Bitmapset* set){
	if (set == NULL)
		return Bitmapset64(0);

    if (set->nwords > 1){
        printf("WARNING: only relids of 64 bits are supported!\n");
    }
	
    return Bitmapset64(set->words[0]);
}

template<typename Bitmapset>
gpuqo_c::Bitmapset* convertBitmapset(Bitmapset set){
	gpuqo_c::Bitmapset *result;

	int nwords = (sizeof(set)*8 + BITS_PER_BITMAPWORD - 1) / BITS_PER_BITMAPWORD;

	result = (gpuqo_c::Bitmapset *) palloc0(bitmapset_size(nwords));
	result->nwords = nwords;

	while (!set.empty()){
		int x = set.lowestPos(); 
		int wordnum = x / BITS_PER_BITMAPWORD;
		int bitnum = x % BITS_PER_BITMAPWORD;

		result->words[wordnum] |= ((gpuqo_c::bitmapword) 1 << bitnum);
		set.unset(x);
	}

	return result;
}

template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* convertGpuqoPlannerInfo(gpuqo_c::GpuqoPlannerInfo *info_c){
	unsigned int size = sizeof(GpuqoPlannerInfo<BitmapsetN>);
	size += sizeof(unsigned int) * info_c->n_fk_selecs;
	size += sizeof(float) * info_c->n_fk_selecs;
	size += sizeof(BitmapsetN) * info_c->n_eq_classes;
	size += sizeof(float) * info_c->n_eq_class_sels;
	size += ceil_div(size, 8)*8; // ceil to 64 bits multiples

	char* p = new char[size];

	GpuqoPlannerInfo<BitmapsetN> *info = (GpuqoPlannerInfo<BitmapsetN>*) p;
	p += sizeof(GpuqoPlannerInfo<BitmapsetN>);

	info->size = size;
	info->n_rels = info_c->n_rels;

	for (int i=0; i < info->n_rels; i++){
		info->edge_table[i] = convertBitmapset<BitmapsetN>(info_c->edge_table[i]);
		info->indexed_edge_table[i] = convertBitmapset<BitmapsetN>(info_c->indexed_edge_table[i]);
	}

	info->n_fk_selecs = info_c->n_fk_selecs;

	unsigned int *fk_selec_idxs = (unsigned int*) p;
	p += sizeof(unsigned int) * info->n_fk_selecs;
	float* fk_selec_sels = (float*) p;
	p += sizeof(float) * info->n_fk_selecs;

	int offset = 0;
	for (int i=0; i < info->n_rels; i++){
		info->base_rels[i].id = convertBitmapset<BitmapsetN>(info_c->base_rels[i].id);
		info->base_rels[i].rows = info_c->base_rels[i].rows;
		info->base_rels[i].cost = BASEREL_COEFF * info_c->base_rels[i].tuples;

		gpuqo_c::FKSelecInfo* fk_selec = info_c->base_rels[i].fk_selecs;
		int j = 0;
		while(fk_selec != NULL){
			fk_selec_idxs[offset+j] = fk_selec->other_baserel;
			fk_selec_sels[offset+j] = fk_selec->sel;
			j++;
			fk_selec = fk_selec->next;
		}
		info->base_rels[i].off_fk_selecs = offset;
		info->base_rels[i].n_fk_selecs = j;
		offset += j;
	}
	info->fk_selec_idxs = fk_selec_idxs;
	info->fk_selec_sels = fk_selec_sels;

	info->n_eq_classes = info_c->n_eq_classes;
	info->n_eq_class_sels = info_c->n_eq_class_sels;

	BitmapsetN* eq_classes = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_classes;
	float* eq_class_sels = (float*) p;
	p += sizeof(float) * info->n_eq_class_sels;

	gpuqo_c::EqClassInfo *ec = info_c->eq_classes;
	int i = 0;
	offset = 0;
	while (ec != NULL){
		eq_classes[i] = convertBitmapset<BitmapsetN>(ec->relids);

		int s = eq_classes[i].size();
		int n = eqClassNSels(s);
		for (int j = 0; j < n; j++)
			eq_class_sels[offset+j] = ec->sels[j];
		
		offset += n;
		i++;
		ec = ec->next;
	}
	info->eq_classes = eq_classes;
	info->eq_class_sels = eq_class_sels;

	return info;
}

template GpuqoPlannerInfo<Bitmapset32>* convertGpuqoPlannerInfo<Bitmapset32>(gpuqo_c::GpuqoPlannerInfo *info_c);
template GpuqoPlannerInfo<Bitmapset64>* convertGpuqoPlannerInfo<Bitmapset64>(gpuqo_c::GpuqoPlannerInfo *info_c);
 
template<typename BitmapsetN>
GpuqoPlannerInfo<BitmapsetN>* copyToDeviceGpuqoPlannerInfo(GpuqoPlannerInfo<BitmapsetN> *info){
	GpuqoPlannerInfo<BitmapsetN> tmp_info = *info;

	char* p;
	
	cudaMalloc(&p, info->size);
	
	GpuqoPlannerInfo<BitmapsetN> *info_gpu = (GpuqoPlannerInfo<BitmapsetN>*) p;
	p += sizeof(GpuqoPlannerInfo<BitmapsetN>);
	
	tmp_info.fk_selec_idxs = (unsigned int*) p;
	p += sizeof(unsigned int) * info->n_fk_selecs;
	cudaMemcpy((void*)tmp_info.fk_selec_idxs, info->fk_selec_idxs, 
			sizeof(unsigned int) * info->n_fk_selecs, cudaMemcpyHostToDevice);
	
	tmp_info.fk_selec_sels = (float*) p;
	p += sizeof(float) * info->n_fk_selecs;
	cudaMemcpy((void*)tmp_info.fk_selec_sels, info->fk_selec_sels, 
		sizeof(float) * info->n_fk_selecs, cudaMemcpyHostToDevice);
	
	tmp_info.eq_classes = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_classes;
	cudaMemcpy((void*)tmp_info.eq_classes, info->eq_classes, 
		sizeof(BitmapsetN) * info->n_eq_classes, cudaMemcpyHostToDevice);
	
	tmp_info.eq_class_sels = (float*) p;
	p += sizeof(float) * info->n_eq_class_sels;
	cudaMemcpy((void*)tmp_info.eq_class_sels, info->eq_class_sels, 
		sizeof(float) * info->n_eq_class_sels, cudaMemcpyHostToDevice);
	
	cudaMemcpy(info_gpu, &tmp_info, sizeof(GpuqoPlannerInfo<BitmapsetN>), cudaMemcpyHostToDevice);

	return info_gpu;
}

template GpuqoPlannerInfo<Bitmapset32>* copyToDeviceGpuqoPlannerInfo<Bitmapset32>(GpuqoPlannerInfo<Bitmapset32> *info);
template GpuqoPlannerInfo<Bitmapset64>* copyToDeviceGpuqoPlannerInfo<Bitmapset64>(GpuqoPlannerInfo<Bitmapset64> *info);

template<typename BitmapsetN>
gpuqo_c::QueryTree* convertQueryTree(QueryTree<BitmapsetN>* qt){
	if (qt == NULL)
		return NULL;
	
	gpuqo_c::QueryTree *result = (gpuqo_c::QueryTree *) palloc(sizeof(gpuqo_c::QueryTree));
	result->id = convertBitmapset<BitmapsetN>(qt->id);
	result->left = convertQueryTree(qt->left);
	result->right = convertQueryTree(qt->right);
	result->cost = qt->cost;
	result->rows = qt->rows;

	free(qt);

	return result;
}

template gpuqo_c::QueryTree* convertQueryTree<Bitmapset32>(QueryTree<Bitmapset32>* qt);
template gpuqo_c::QueryTree* convertQueryTree<Bitmapset64>(QueryTree<Bitmapset64>* qt);
  