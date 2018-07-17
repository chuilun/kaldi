

all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-pdf-prior.o nnet-randomizer.o \
	   	   nnet-example.o nnet-model-sync.o \
           nnet-compute-sequential-parallel.o nnet-compute-parallel.o \
           nnet-affine-preconditioned-transform.o nnet-model-merge-function.o \
	       nnet-compute-lstm-parallel.o nnet-compute-lstm-asgd.o \
	       nnet-compute-forward.o nnet-compute-ctc-parallel.o  \
		   nnet-compute-lstm-lm-parallel.o 

LIBNAME = kaldi-nnet0

ADDLIBS = ../hmm/kaldi-hmm.a ../nnet2/kaldi-nnet2.a ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../thread/kaldi-thread.a ../lat/kaldi-lat.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../util/kaldi-util.a

include ../makefiles/default_rules.mk

