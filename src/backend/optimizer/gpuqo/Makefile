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
NVOBJS = gpuqo_dpsize.o

# NVIDIA-specific code (do not edit)

NVLINK = dlink.o
OBJS += $(NVLINK)
OBJS += $(NVOBJS)

NVCC := $(cuda_path)/bin/nvcc
NVCCFLAGS = $(CPPFLAGS)

ifeq ($(enable_debug),yes)
NVCCFLAGS += -O0 -g -DGPUQO_DEBUG
else
NVCCFLAGS += -O2 
endif

ifeq ($(enable_profile),yes)
NVCCFLAGS += -DGPUQO_PROFILE
endif

GENCODE_FLAGS = $(foreach sm, $(with_cudasm), -gencode arch=compute_$(sm),code=sm_$(sm))
BUILDDIR = $(top_builddir)/$(subdir)

$(NVLINK): $(NVOBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

$(NVOBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(GENCODE_FLAGS) 


include $(top_srcdir)/src/backend/common.mk
