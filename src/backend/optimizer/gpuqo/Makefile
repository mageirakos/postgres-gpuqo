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

OBJS   = gpuqo_main.o 
NVOBJS = gpuqo_kernel.o

# NVIDIA-specific code (do not edit)

NVLINK = dlink.o
OBJS += $(NVLINK)
OBJS += $(NVOBJS)

SM := 35
NVCC := $(cuda_path)/bin/nvcc
NVCCFLAGS = -O2
GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)
BUILDDIR = $(top_builddir)/$(subdir)

$(NVLINK): $(NVOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(NVOBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(GENCODE_FLAGS) 


include $(top_srcdir)/src/backend/common.mk
