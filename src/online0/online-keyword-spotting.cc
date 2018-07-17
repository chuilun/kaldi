// online0/online-keyword-spotting.cc
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

#include "online0/online-keyword-spotting.h"

namespace kaldi {

OnlineKeywordSpotting::OnlineKeywordSpotting(std::string cfg) :
		feature_opts_(NULL), forward_opts_(NULL),
		feature_pipeline_(NULL), forward_(NULL),
		state_(FEAT_START), score_(0.0), iswakeup_(0), len_(0), sample_offset_(0), frame_ready_(0), frame_offset_(0),
		post_offset_(0), wakeup_frame_(0) {

	// main config
	kws_config_ = new OnlineKeywordSpottingConfig;
	ReadConfigFromFile(cfg, kws_config_);

	feature_opts_ = new OnlineNnetFeaturePipelineOptions(kws_config_->feature_cfg);
	// neural network forward options
	forward_opts_ = new OnlineNnetForwardOptions;
	ReadConfigFromFile(kws_config_->forward_cfg, forward_opts_);

	//keywords id list
    std::vector<std::string> kws_str;
    kaldi::SplitStringToVector(kws_config_->keywords_id, "|", false, &kws_str);
    keywords_.resize(kws_str.size());
    for(int i = 0; i < kws_str.size(); i++) {
            if (!kaldi::SplitStringToIntegers(kws_str[i], ":", false, &keywords_[i]))
                    KALDI_ERR << "Invalid keywords id string " << kws_str[i];
    }   
}

void OnlineKeywordSpotting::InitKws() {
	// base feature pipeline
	feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);
	// forward
	forward_ = new OnlineNnetForward(*forward_opts_);

	Reset();
}

int OnlineKeywordSpotting::FeedData(void *data, int nbytes, FeatState state) {

	if (nbytes > 0) {
		int size = nbytes/sizeof(float);
		wav_buffer_.Resize(size, kUndefined);
		memcpy((char*)(wav_buffer_.Data()), (char*)data, nbytes);
		len_ += size;
	}

	int num_frames;
	state_ = state;
	if (nbytes >= 0) {
		feature_pipeline_->AcceptWaveform(feature_opts_->samp_freq, wav_buffer_);

		if (state == FEAT_END)
			feature_pipeline_->InputFinished();

		frame_ready_ = feature_pipeline_->NumFramesReady();

		if (frame_ready_ == frame_offset_)
			return 0;

        feat_in_.Resize(frame_ready_-frame_offset_, forward_->InputDim());
		for (int i = frame_offset_; i < frame_ready_; i++) {
			feature_pipeline_->GetFrame(i, &feat_in_.Row(i-frame_offset_));
		}

		// feed forward to neural network
		forward_->Forward(feat_in_, &nnet_out_);

		int new_rows = nnet_out_.NumRows();
		Resize(posterior_, frame_offset_, new_rows);
		posterior_.RowRange(frame_offset_, new_rows).CopyFromMat(nnet_out_);

		num_frames = frame_ready_ - frame_offset_;
		frame_offset_ = frame_ready_;
	}
	return num_frames;
}

void OnlineKeywordSpotting::Resize(Matrix<BaseFloat> &mat, int valid_rows, int new_rows) {
	if (mat.NumRows() < valid_rows + new_rows) {
		int step = new_rows > MATRIX_INC_STEP ? new_rows : MATRIX_INC_STEP;
	    	Matrix<BaseFloat> tmp(mat.NumRows()+step, mat.NumCols(), kUndefined);
	    	if (valid_rows > 0)
	    		tmp.RowRange(0, valid_rows).CopyFromMat(mat.RowRange(0, valid_rows));
	    	mat.Swap(&tmp);
	}
}

int OnlineKeywordSpotting::isWakeUp() {
	int new_rows = frame_offset_ - post_offset_;

	if (new_rows <= 0)
		return iswakeup_;

	Resize(post_smooth_, post_offset_, new_rows);
	Resize(confidence_, post_offset_, new_rows);

	int w_smooth = kws_config_->smooth_window;
	int w_max = kws_config_->sliding_window;
	int word_interval = kws_config_->word_interval;
	int cols = keywords_.size()+1;

	int hs, hm, pre_t, cur_t;
	float sum, mul, sscore, pre_score;

	// posterior smoothing
	for (int j = post_offset_; j < post_offset_+new_rows; j++) {
		for (int i = 1; i < cols; i++) {
			hs = j-w_smooth+1 > 0 ? j-w_smooth+1 : 0;
			sum = 0;
			for (int k = hs; k <= j; k++) {
				for (int m = 0; m < keywords_[i-1].size(); m++)
				  sum += posterior_(k, keywords_[i-1][m]);
			}
			post_smooth_(j, i) = sum/(j-hs+1);
		}
	}

	// compute confidence score
	// confidence.Set(1.0);
	for (int j = post_offset_; j < post_offset_+new_rows; j++) {
		mul = 1.0;
		hm = j-w_max+1 > 0 ? j-w_max+1 : 0;

		// the first keyword
		for (int k = 1; k <= j-hm+1; k++) {
			buffer_(k, 1) = post_smooth_(k+hm-1,1);
			buffer_(k, 2) = k; // time stamp
			if (buffer_(k, 1) < buffer_(k-1, 1)) {
				buffer_(k, 1) = buffer_(k-1, 1);
				buffer_(k, 2) = buffer_(k-1, 2);
			}
		}

		// 2,...,n keywords
		for (int i = 2; i < cols; i++) {
			for (int k = i; k <= j-hm+1; k++) {
				buffer_(k, 2*i-1) = buffer_(k-1, 2*i-1);
				buffer_(k, 2*i) = buffer_(k-1, 2*i);
				sscore = buffer_(k-1, 2*i-3) * post_smooth_(k+hm-1, i);
				if (buffer_(k, 2*i-1) < sscore) {
					buffer_(k, 2*i-1) = sscore;
					buffer_(k, 2*i) = k-1; //(k-1)+(hm-1);
				}
			}
		}

        // final score
		mul = buffer_(j-hm+1, 2*(cols-1)-1);
		confidence_(j,0) = pow(mul, 1.0/(cols-1));
		confidence_(j,1) = j; // time stamp

		// back tracking
		cur_t = j-hm+1;
		for (int i = cols-1; i > 1; i--) {
			pre_t = buffer_(cur_t, 2*i);
			pre_score = buffer_(pre_t, 2*(i-1)-1);

			// the nth keyword score
			confidence_(j,2*i) = mul/pre_score;

			// the nth keyword time stamp
			confidence_(j,2*i+1) = (pre_t+1)+(hm-1);

			mul = pre_score;
			cur_t = pre_t;
		}

		confidence_(j,2) = buffer_(cur_t, 1);
		confidence_(j,3) = buffer_(cur_t, 2)+(hm-1);

		// is wakeup?
		bool flag = true;
		int interval;
		for (int i = 2; i < cols; i++) {
			interval = confidence_(j,2*i+1)-confidence_(j,2*(i-1)+1);
			if (interval >= word_interval || interval <= 0) {
				flag = false;
				break;
			}
		}

		if (score_ < confidence_(j,0)) {
			score_ = confidence_(j,0);
			wakeup_frame_ = j;
		}

		if (confidence_(j,0) >= kws_config_->wakeup_threshold && flag) {
			iswakeup_ = 1;
		}
	}

	post_offset_ += new_rows;

	// maximal score statistical information
	if (state_ == FEAT_END) {
		std::ostringstream os;
		os << wakeup_frame_ << " ";
		for (int i = 0; i < cols; i++) {
			os << confidence_(wakeup_frame_,2*i) << " ";
		}
		os << confidence_(wakeup_frame_, 3) << " ";
		for (int i = 2; i < cols; i++) {
			os << confidence_(wakeup_frame_,2*i+1)-confidence_(wakeup_frame_,2*(i-1)+1) << "\t";
		}
		os << std::endl;
		KALDI_VLOG(1) << os.str();
	}

	return iswakeup_;
}

void OnlineKeywordSpotting::Reset() {
	feature_pipeline_->Reset();
	forward_->ResetHistory();

	posterior_.Resize(MATRIX_INC_STEP, forward_->OutputDim());
	int cols = keywords_.size()+1;
	post_smooth_.Resize(MATRIX_INC_STEP, cols);
	confidence_.Resize(MATRIX_INC_STEP, 2*cols);
	buffer_.Resize(kws_config_->sliding_window+1, keywords_.size()*2+1);

	iswakeup_ = len_ = sample_offset_ = 0;
	frame_ready_ = frame_offset_ = post_offset_ = 0;
	wakeup_frame_ = 0;
	score_ = 0.0;
	state_ = FEAT_START;
}

void OnlineKeywordSpotting::Destory() {
	if (feature_pipeline_ != NULL) {
		delete feature_opts_; feature_opts_ = NULL;
		delete forward_opts_; forward_opts_ = NULL;
		delete feature_pipeline_;	  feature_pipeline_ = NULL;
		delete forward_;	forward_= NULL;
		delete kws_config_;	kws_config_ = NULL;
	}
}

}	// namespace kaldi




