#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for the GPU query optimizer module
#
# src/backend/optimizer/gpuqo/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/optimizer/gpuqo
top_builddir = ../../../..
include $(top_builddir)/src/Makefile.global

OBJS   = gpuqo_main.o gpuqo_bfs_indexing.o
NVOBJS = gpuqo_dpsize.o gpuqo_dpsub_common.o gpuqo_dpsub_unfiltered.o gpuqo_dpsub_filtered.o gpuqo_dpsub_csg.o  gpuqo_dpsub_tree.o gpuqo_debug.o gpuqo_cost.o gpuqo_joinrelation.o gpuqo_query_tree.o gpuqo_binomial.o gpuqo_cpu_common.o gpuqo_cpu_sequential.o gpuqo_cpu_dpe.o gpuqo_dependency_buffer.o gpuqo_cpu_dpsize.o gpuqo_cpu_dpsub.o gpuqo_cpu_dpccp.o gpuqo_malloc.o gpuqo_hashtable.o

# NVIDIA-specific code (do not edit)

NVLINK = dlink.o
OBJS += $(NVLINK)
OBJS += $(NVOBJS)

NVCC := $(cuda_path)/bin/nvcc
NVCCFLAGS = $(CPPFLAGS)

ifeq ($(enable_debug),yes)
NVCCFLAGS += -g -lineinfo
endif

ifneq (,$(findstring -O0,$(CFLAGS)))
    NVCCFLAGS += -O0
else ifneq (,$(findstring -O1,$(CFLAGS)))
    NVCCFLAGS += -O1
else ifneq (,$(findstring -O2,$(CFLAGS)))
    NVCCFLAGS += -O2
else ifneq (,$(findstring -O3,$(CFLAGS)))
    NVCCFLAGS += -O3
else
	NVCCFLAGS += -O2
endif

ifeq ($(enable_gpuqo_profiling),yes)
NVCCFLAGS += -DGPUQO_PROFILE
endif

ifeq ($(enable_gpuqo_kernel_profiling),yes)
NVCCFLAGS += -DGPUQO_KERNEL_PROFILE
endif

ifeq ($(enable_gpuqo_debug),yes)
NVCCFLAGS += -DGPUQO_DEBUG
endif

GENCODE_FLAGS = $(foreach sm, $(with_cudasm), -gencode arch=compute_$(sm),code=sm_$(sm))
BUILDDIR = $(top_builddir)/$(subdir)

$(NVLINK): $(NVOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(NVOBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 


include $(top_srcdir)/src/backend/common.mk
