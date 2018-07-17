// online0/Online-fst-decoder.cc
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

#include "online0/online-util.h"
#include "online0/online-fst-decoder.h"

namespace kaldi {

OnlineFstDecoder::OnlineFstDecoder(std::string cfg) :
		decoder_opts_(NULL), forward_opts_(NULL), feature_opts_(NULL), decoding_opts_(NULL),
		decode_fst_(NULL), word_syms_(NULL), decodable_(NULL),
		decoder_(NULL), decoding_(NULL), decoder_thread_(NULL),
		feature_pipeline_(NULL), forward_(NULL),
		words_writer_(NULL), alignment_writer_(NULL), state_(FEAT_START),
		len_(0), sample_offset_(0), frame_offset_(0), frame_ready_(0),
		in_skip_(0), out_skip_(0), chunk_length_(0), cur_result_idx_(0) {

	// main config
	decoding_opts_ = new OnlineNnetDecodingOptions;
	ReadConfigFromFile(cfg, decoding_opts_);

	// decoder feature forward config
	decoder_opts_ = new OnlineNnetFasterDecoderOptions;
	forward_opts_ = new OnlineNnetForwardOptions;
	feature_opts_ = new OnlineNnetFeaturePipelineOptions(decoding_opts_->feature_cfg);
	ReadConfigFromFile(decoding_opts_->decoder_cfg, decoder_opts_);
	ReadConfigFromFile(decoding_opts_->forward_cfg, forward_opts_);
}

void OnlineFstDecoder::Destory() {
	Abort();
	if (decoder_opts_ != NULL) {
		delete decoder_opts_;	decoder_opts_ = NULL;
		delete forward_opts_;	forward_opts_ = NULL;
		delete feature_opts_;	feature_opts_ = NULL;
		delete decoding_opts_;	decoding_opts_ = NULL;
	}

	if (decode_fst_ != NULL) {
		delete decode_fst_;	decode_fst_ = NULL;
		delete word_syms_;	word_syms_ = NULL;
		delete decodable_;	decodable_ = NULL;
	}

	if (words_writer_ != NULL) {
		delete words_writer_;	words_writer_ = NULL;
	}

	if (alignment_writer_ != NULL) {
		delete alignment_writer_; alignment_writer_ = NULL;
	}

	if (decoder_ != NULL) {
		delete decoder_;	decoder_ = NULL;
		delete decoding_;	decoding_ = NULL;
		delete decoder_thread_;		decoder_thread_ = NULL;
		delete feature_pipeline_;	feature_pipeline_ = NULL;
		delete forward_;			forward_ = NULL;
	}
}

void OnlineFstDecoder::InitDecoder() {
#if HAVE_CUDA==1
    if (forward_opts_->use_gpu == "yes")
        CuDevice::Instantiate().Initialize();
#endif
	// trainsition model
	bool binary;
	if (decoding_opts_->model_rspecifier != "") {
		Input ki(decoding_opts_->model_rspecifier, &binary);
		trans_model_.Read(ki.Stream(), binary);
	}

	// HCLG fst graph
	decode_fst_ = fst::ReadFstKaldi(decoding_opts_->fst_rspecifier);
	if (!(word_syms_ = fst::SymbolTable::ReadText(decoding_opts_->word_syms_filename)))
		KALDI_ERR << "Could not read symbol table from file " << decoding_opts_->word_syms_filename;

	// decodable feature pipe to decoder
	if (decoding_opts_->model_rspecifier != "")
		decodable_ = new OnlineDecodableMatrixMapped(trans_model_, decoding_opts_->acoustic_scale);
	else
		decodable_ = new OnlineDecodableMatrixCtc(decoding_opts_->acoustic_scale);

	if (decoding_opts_->words_wspecifier != "")
		words_writer_ = new Int32VectorWriter(decoding_opts_->words_wspecifier);
	if (decoding_opts_->alignment_wspecifier != "")
		alignment_writer_ = new Int32VectorWriter(decoding_opts_->alignment_wspecifier);

	// decoder
	decoder_ = new OnlineNnetFasterDecoder(*decode_fst_, *decoder_opts_);
	decoding_ = new OnlineNnetDecodingClass(*decoding_opts_,
		    								decoder_, decodable_, &decoder_sync_,
											&result_);
	decoder_thread_ = new MultiThreader<OnlineNnetDecodingClass>(1, *decoding_);

	// feature pipeline
	feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	// forward
	forward_ = new OnlineNnetForward(*forward_opts_);

	// decoding buffer
	in_skip_ = decoding_opts_->skip_inner ? 1:decoding_opts_->skip_frames,
	        out_skip_ = decoding_opts_->skip_inner ? decoding_opts_->skip_frames : 1;
	int feat_dim = feature_pipeline_->Dim();
	feat_in_.Resize(out_skip_*forward_opts_->batch_size, feat_dim);
	// wav buffer
	wav_buffer_.Resize(VECTOR_INC_STEP, kSetZero); // 16k, 10s

	if (decoding_opts_->chunk_length_secs > 0) {
		chunk_length_ = int32(feature_opts_->samp_freq * decoding_opts_->chunk_length_secs);
		if (chunk_length_ == 0) chunk_length_ = 1;
	} else {
		chunk_length_ = std::numeric_limits<int32>::max();
	}
}

void OnlineFstDecoder::Reset() {
	feature_pipeline_->Reset();
	forward_->ResetHistory();
	decodable_->Reset();
	result_.clear();
	len_ = 0;
	sample_offset_ = 0;
	frame_offset_ = 0;
	frame_ready_ = 0;
	cur_result_idx_ = 0;
	state_ = FEAT_START;
	wav_buffer_.Resize(VECTOR_INC_STEP, kUndefined); // 16k, 10s
}

void OnlineFstDecoder::FeedData(void *data, int nbytes, FeatState state) {
	// extend buffer
	if (wav_buffer_.Dim() < len_+nbytes/sizeof(float)) {
		Vector<BaseFloat> tmp(wav_buffer_.Dim()+VECTOR_INC_STEP, kUndefined);
		memcpy((char*)tmp.Data(), (char*)wav_buffer_.Data(), len_*sizeof(float));
		wav_buffer_.Swap(&tmp);
	}

	BaseFloat *wav_data = wav_buffer_.Data();
	if (nbytes > 0) {
		memcpy((char*)(wav_data+len_), (char*)data, nbytes);
		len_ += nbytes/sizeof(float);
	}

	int32 samp_remaining = len_ - sample_offset_;
	int32 batch_size = forward_opts_->batch_size * decoding_opts_->skip_frames;

	if (sample_offset_ <= len_) {
		SubVector<BaseFloat> wave_part(wav_buffer_, sample_offset_, samp_remaining);
		feature_pipeline_->AcceptWaveform(feature_opts_->samp_freq, wave_part);
		sample_offset_ += samp_remaining;

		if (state == FEAT_END)
			feature_pipeline_->InputFinished();

		while (true) {
			frame_ready_ = feature_pipeline_->NumFramesReady();
			if (!feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ < frame_offset_+batch_size)
				break;
            else if (feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ == frame_offset_)
                break;
			else if (feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ < frame_offset_+batch_size) {
				frame_ready_ -= frame_offset_;
				feat_in_.SetZero();
			}
			else
				frame_ready_ = batch_size;

			for (int i = 0; i < frame_ready_; i += in_skip_) {
				feature_pipeline_->GetFrame(frame_offset_+i, &feat_in_.Row(i/in_skip_));
			}

			frame_offset_ += frame_ready_;
			// feed forward to neural network
			forward_->Forward(feat_in_, &feat_out_);

			// copy posterior
			if (decoding_opts_->copy_posterior) {
				feat_out_ready_.Resize(frame_ready_, feat_out_.NumCols(), kUndefined);
				for (int i = 0; i < frame_ready_; i++)
					feat_out_ready_.Row(i).CopyFromVec(feat_out_.Row(i/decoding_opts_->skip_frames));

				decodable_->AcceptLoglikes(&feat_out_ready_);
			}
			else
				decodable_->AcceptLoglikes(&feat_out_);

			// wake up decoder thread
			result_.num_frames += frame_ready_;
			decoder_sync_.DecoderSignal();
		}
	}

	if (state == FEAT_END) {
		decodable_->InputIsFinished();
		decoder_sync_.DecoderSignal();

		// waiting a utterance finished
		decoder_sync_.UtteranceWait();
	}
}

Result* OnlineFstDecoder::GetResult(FeatState state) {
	std::vector<int32> word_ids;
	int size = result_.word_ids_.size();
	for (int i = cur_result_idx_; i < size; i++)
		word_ids.push_back(result_.word_ids_[i]);

    bool newutt = (state == FEAT_END);
	PrintPartialResult(word_ids, word_syms_, newutt);
    std::cout.flush();

	cur_result_idx_ = size;
	return &result_;
}

void OnlineFstDecoder::Abort() {
	decoder_sync_.Abort();
	sleep(0.1);
	decoder_sync_.DecoderSignal();
}

}	// namespace kaldi




