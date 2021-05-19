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
	size += sizeof(BitmapsetN) * info_c->n_eq_classes;
	size += sizeof(float) * info_c->n_eq_class_sels;
	size += sizeof(BitmapsetN) * info_c->n_eq_class_fks;
	size += ceil_div(size, 8)*8; // ceil to 64 bits multiples

	char* p = new char[size];

	GpuqoPlannerInfo<BitmapsetN> *info = (GpuqoPlannerInfo<BitmapsetN>*) p;
	p += sizeof(GpuqoPlannerInfo<BitmapsetN>);

	info->size = size;
	info->n_rels = info_c->n_rels;
	info->n_iters = info_c->n_rels; // will be overwritten by IDP

	for (int i=0; i < info->n_rels; i++){
		info->edge_table[i] = convertBitmapset<BitmapsetN>(info_c->edge_table[i]);
		info->indexed_edge_table[i] = convertBitmapset<BitmapsetN>(info_c->indexed_edge_table[i]);
	}

	for (int i=0; i < info->n_rels; i++){
		info->base_rels[i].id = convertBitmapset<BitmapsetN>(info_c->base_rels[i].id);
		info->base_rels[i].rows = info_c->base_rels[i].rows;
		info->base_rels[i].tuples = info_c->base_rels[i].tuples;
		info->base_rels[i].cost = BASEREL_COEFF * info_c->base_rels[i].tuples;
		info->base_rels[i].composite = false;
	}

	info->n_eq_classes = info_c->n_eq_classes;
	info->n_eq_class_sels = info_c->n_eq_class_sels;
	info->n_eq_class_fks = info_c->n_eq_class_fks;

	BitmapsetN* eq_classes = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_classes;
	float* eq_class_sels = (float*) p;
	p += sizeof(float) * info->n_eq_class_sels;
	BitmapsetN* eq_class_fk = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_class_fks;

	gpuqo_c::EqClassInfo *ec = info_c->eq_classes;
	int i = 0;
	int offset_sels = 0;
	int offset_fks = 0;
	while (ec != NULL){
		eq_classes[i] = convertBitmapset<BitmapsetN>(ec->relids);

		int s = eq_classes[i].size();
		int n = eqClassNSels(s);
		for (int j = 0; j < n; j++)
			eq_class_sels[offset_sels+j] = ec->sels[j];

		for (int j = 0; j < s; j++)
			eq_class_fk[offset_fks+j] = convertBitmapset<BitmapsetN>(ec->fk[j]);
		
		offset_sels += n;
		offset_fks += s;
		i++;
		ec = ec->next;
	}
	info->eq_classes = eq_classes;
	info->eq_class_sels = eq_class_sels;
	info->eq_class_fk = eq_class_fk;

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
	
	tmp_info.eq_classes = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_classes;
	cudaMemcpy((void*)tmp_info.eq_classes, info->eq_classes, 
		sizeof(BitmapsetN) * info->n_eq_classes, cudaMemcpyHostToDevice);
	
	tmp_info.eq_class_sels = (float*) p;
	p += sizeof(float) * info->n_eq_class_sels;
	cudaMemcpy((void*)tmp_info.eq_class_sels, info->eq_class_sels, 
		sizeof(float) * info->n_eq_class_sels, cudaMemcpyHostToDevice);
	
	tmp_info.eq_class_fk = (BitmapsetN*) p;
	p += sizeof(BitmapsetN) * info->n_eq_class_fks;
	cudaMemcpy((void*)tmp_info.eq_class_fk, info->eq_class_fk, 
		sizeof(BitmapsetN) * info->n_eq_class_fks, cudaMemcpyHostToDevice);
	
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
  