// online0/online-keyword-spotting.h
// Copyright 2017-2018   Shanghai Jiao Tong University (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_KEYWORD_SPOTTING_H_
#define ONLINE0_ONLINE_KEYWORD_SPOTTING_H_

#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet-forward.h"

namespace kaldi {

struct OnlineKeywordSpottingConfig {

	/// feature pipeline config
	OnlineNnetFeaturePipelineConfig feature_cfg;
	/// neural network forward config
	std::string forward_cfg;

	int32 smooth_window;
	int32 sliding_window;
	int32 word_interval;
	std::string keywords_id;
	BaseFloat wakeup_threshold;

	OnlineKeywordSpottingConfig(): smooth_window(10), sliding_window(80),
								keywords_id("348|363:369|328|355:349"), wakeup_threshold(0.4){ }

	void Register(OptionsItf *opts) {
		feature_cfg.Register(opts);
		opts->Register("forward-config", &forward_cfg, "Configuration file for neural network forward");
		opts->Register("word-interval", &word_interval, "Word interval between each keyword");
		opts->Register("smooth-window", &smooth_window, "Smooth the posteriors over a fixed time window of size smooth-window.");
		opts->Register("sliding-window", &sliding_window, "The confidence score is computed within a sliding window of sliding-window.");
		opts->Register("keywords-id", &keywords_id, "keywords index in network output(e.g. 348|363:369|328|355:349.");
		opts->Register("wakeup-threshold", &wakeup_threshold, "Greater or equal this threshold will be wakeup.");
	}
};

typedef enum {
	FEAT_START,
	FEAT_APPEND,
	FEAT_END,
}FeatState;

class OnlineKeywordSpotting {

public:
	OnlineKeywordSpotting(std::string cfg);
	virtual ~OnlineKeywordSpotting() { Destory(); }

	// initialize Kws
	void InitKws();

	// feed wave data to Kws
	int FeedData(void *data, int nbytes, FeatState state);

	// is wakup currently?
	int isWakeUp();

	// Reset Kws
	void Reset();

private:
	void Destory();
	void Resize(Matrix<BaseFloat> &mat, int valid_rows, int new_rows);

	const static int MATRIX_INC_STEP = 1024;
	// options
	OnlineKeywordSpottingConfig *kws_config_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	OnlineNnetForwardOptions *forward_opts_;

	// feature pipeline
	OnlineNnetFeaturePipeline *feature_pipeline_;
	// forward
	OnlineNnetForward *forward_;

	std::vector<std::vector<int32> > keywords_;

	// decoding buffer
	Matrix<BaseFloat> feat_in_, nnet_out_, posterior_,
						post_smooth_, confidence_, buffer_;
	// wav buffer
	Vector<BaseFloat> wav_buffer_;
	FeatState state_;
	float score_;
	int iswakeup_, len_, sample_offset_, frame_ready_, frame_offset_, post_offset_, wakeup_frame_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_KEYWORD_SPOTTING_H_ */
