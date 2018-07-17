// online0/Online-fst-decoder.h
// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef ONLINE0_ONLINE_FST_DECODER_H_
#define ONLINE0_ONLINE_FST_DECODER_H_

#include "fstext/fstext-lib.h"
#include "decoder/decodable-matrix.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"

#include "online0/online-nnet-faster-decoder.h"
#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet-forward.h"
#include "online0/online-nnet-decoding.h"

namespace kaldi {

typedef enum {
	FEAT_START,
	FEAT_APPEND,
	FEAT_END,
}FeatState;

class OnlineFstDecoder {

public:
	OnlineFstDecoder(std::string cfg);
	virtual ~OnlineFstDecoder() { Destory(); }

	// initialize decoder
	void InitDecoder();

	// feed wave data to decoder
	void FeedData(void *data, int nbytes, FeatState state);

	// get online decoder result
	Result* GetResult(FeatState state);

	// Reset decoder
	void Reset();

	// abort decoder
	void Abort();

private:
	void Destory();
	const static int VECTOR_INC_STEP = 16000*10;

	OnlineNnetFasterDecoderOptions *decoder_opts_;
	OnlineNnetForwardOptions *forward_opts_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;

	OnlineNnetDecodingOptions *decoding_opts_;

	TransitionModel trans_model_;
	fst::Fst<fst::StdArc> *decode_fst_;
	fst::SymbolTable *word_syms_;
	OnlineDecodableInterface *decodable_;

	// decoder
	DecoderSync decoder_sync_;
	OnlineNnetFasterDecoder *decoder_;
	OnlineNnetDecodingClass *decoding_;
	MultiThreader<OnlineNnetDecodingClass> *decoder_thread_;

	// feature pipeline
	OnlineNnetFeaturePipeline *feature_pipeline_;
	// forward
	OnlineNnetForward *forward_;

	// decode result
	Int32VectorWriter *words_writer_;
	Int32VectorWriter *alignment_writer_;

	Result result_;
	FeatState state_;

	// decoding buffer
	Matrix<BaseFloat> feat_in_, feat_out_, feat_out_ready_;
	// wav buffer
	Vector<BaseFloat> wav_buffer_;
	// online feed
	int len_, sample_offset_, frame_offset_, frame_ready_;
	int in_skip_, out_skip_, chunk_length_, cur_result_idx_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_FST_DECODER_H_ */
