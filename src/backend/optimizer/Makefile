#
# Makefile for optimizer
#
# src/backend/optimizer/Makefile
#

subdir = src/backend/optimizer
top_builddir = ../../..
include $(top_builddir)/src/Makefile.global

SUBDIRS     = geqo path plan prep util

ifeq ($(enable_cuda),yes)
SUBDIRS += gpuqo
endif

include $(top_srcdir)/src/backend/common.mk
