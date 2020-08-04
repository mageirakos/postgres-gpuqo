#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for optimizer/path
#
# IDENTIFICATION
#    src/backend/optimizer/path/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/optimizer/path
top_builddir = ../../../..
include $(top_builddir)/src/Makefile.global

OBJS = allpaths.o clausesel.o costsize.o equivclass.o indxpath.o \
       joinpath.o joinrels.o pathkeys.o tidpath.o

ifeq ($(enable_cuda),yes)
override CFLAGS += -DENABLE_GPUQO
endif

include $(top_srcdir)/src/backend/common.mk
