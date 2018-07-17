
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = pid-test online-nnet-ipc-decoder  online-nnet-ipc-forward \
		   online-nnet-wav-decoder  online-feature-extractor online-decoder-test \
		   online-fast-decoder online-kws-test 

OBJFILES =

TESTFILES =

ADDLIBS = ../online0/kaldi-online0.a ../nnet0/kaldi-nnet0.a ../nnet2/kaldi-nnet2.a\
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../thread/kaldi-thread.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a 
include ../makefiles/default_rules.mk
