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

#include "optimizer/paths.h"
#include "optimizer/gpuqo.h"

#define N 256
#define MIN 0 
#define MAX 1000

void initialize_matrices(float * a, float * b);

/*
 * gpuqo
 *	  solution of the query optimization problem
 *	  using GPU acceleration (CUDA)
 */

RelOptInfo *
gpuqo(PlannerInfo *root, int number_of_rels, List *initial_rels)
{
	float * a = (float *)malloc(N * N * N * sizeof(float));
    float * b = (float *)malloc(N * N * N * sizeof(float));

	printf("Hello, here is gpuqo!\n"
           "My creator did not implement it yet so I will just call a random "
		   "CUDA kernel and then execute the standard join search\n");

    initialize_matrices(a, b);
    perform_stencil(a, b, N);

	return standard_join_search(root, number_of_rels, initial_rels);
}

void initialize_matrices(float * a, float * b) {
    for (int i = 0; i < N * N * N; i ++) {
        a[i] = 0.0;
        b[i] = MIN + (MAX - MIN) * (rand() / (float)RAND_MAX);
    }
}