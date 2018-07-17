// online0/online-nnet-feature-pipeline.cc

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

#include "online0/online-nnet-feature-pipeline.h"

namespace kaldi {

OnlineNnetFeaturePipelineOptions::OnlineNnetFeaturePipelineOptions(
		const OnlineNnetFeaturePipelineConfig &config):
		feature_type("fbank"), add_pitch(false),
		add_cmvn(false), add_deltas(false), splice_feats(false), samp_freq(16000) {

	if (config.feature_type == "mfcc" || config.feature_type == "plp" ||
	  config.feature_type == "fbank") {
		feature_type = config.feature_type;
	} else {
		KALDI_ERR << "Invalid feature type: " << config.feature_type << ". "
			  << "Supported feature types: mfcc, plp, fbank.";
	}

	if (config.mfcc_config != "") {
		ReadConfigFromFile(config.mfcc_config, &mfcc_opts);
		samp_freq = mfcc_opts.frame_opts.samp_freq;
		if (feature_type != "mfcc")
			KALDI_WARN << "--mfcc-config option has no effect "
				 << "since feature type is set to " << feature_type << ".";
	}  // else use the defaults.

	if (config.plp_config != "") {
		ReadConfigFromFile(config.plp_config, &plp_opts);
		samp_freq = plp_opts.frame_opts.samp_freq;
		if (feature_type != "plp")
			KALDI_WARN << "--plp-config option has no effect "
				 << "since feature type is set to " << feature_type << ".";
	}  // else use the defaults.

	if (config.fbank_config != "") {
		ReadConfigFromFile(config.fbank_config, &fbank_opts);
		samp_freq = fbank_opts.frame_opts.samp_freq;
		if (feature_type != "fbank")
			KALDI_WARN << "--fbank-config option has no effect "
				 << "since feature type is set to " << feature_type << ".";
	}  // else use the defaults.

    add_cmvn = config.add_cmvn;
	if (config.cmvn_config != "") {
		ReadConfigFromFile(config.cmvn_config, &cmvn_opts);
		if (!add_cmvn)
			KALDI_WARN << "--cmvn-config option has no effect "
				 << "since you did not supply --add-cmvn option.";
	}  // else use the defaults.

    add_deltas = config.add_deltas;
	if (config.delta_config != "") {
		ReadConfigFromFile(config.delta_config, &delta_opts);
		if (!add_deltas)
			KALDI_WARN << "--delta-config option has no effect "
				 << "since you did not supply --add-deltas option.";
	}  // else use the defaults.

    splice_feats = config.splice_feats;
	if (config.splice_config != "") {
		ReadConfigFromFile(config.splice_config, &splice_opts);
		if (!splice_feats)
			KALDI_WARN << "--delta-splice option has no effect "
				 << "since you did not supply --add-splice option.";
	}  // else use the defaults.
}

///
OnlineNnetFeaturePipeline::OnlineNnetFeaturePipeline(
    const OnlineNnetFeaturePipelineOptions &opts):
		opts_(opts) {
  if (opts.feature_type == "mfcc") {
    base_feature_ = new OnlineStreamMfcc(opts.mfcc_opts);
  } else if (opts.feature_type == "plp") {
    base_feature_ = new OnlineStreamPlp(opts.plp_opts);
  } else if (opts.feature_type == "fbank") {
    base_feature_ = new OnlineStreamFbank(opts.fbank_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << opts.feature_type;
  }

  final_feature_ = base_feature_;

  /// online cmvn feature
  if (opts.add_cmvn) {
	cmvn_feature_ = new OnlineStreamCmvnFeature(opts.cmvn_opts, base_feature_);
	final_feature_ = cmvn_feature_;
  } else {
	cmvn_feature_ = NULL;
  }

  /// add deltas feature
  if (opts.add_deltas) {
	delta_feature_ = new OnlineStreamDeltaFeature(opts.delta_opts, final_feature_);
	final_feature_ = delta_feature_;
  } else {
	delta_feature_ = NULL;
  }

  /// add splice feature
  if (opts.splice_feats) {
	  splice_feature_ = new OnlineStreamSpliceFeature(opts.splice_opts, final_feature_);
	  final_feature_ = splice_feature_;
  } else {
	  splice_feature_ = NULL;
  }
}

OnlineNnetFeaturePipeline::~OnlineNnetFeaturePipeline() {
  // Note: the delete command only deletes pointers that are non-NULL.  Not all
  // of the pointers below will be non-NULL.
  // Some of the online-feature pointers are just copies of other pointers,
  // and we do have to avoid deleting them in those cases.
	if (splice_feature_ != NULL)
		delete splice_feature_;
	if (delta_feature_ != NULL)
		delete delta_feature_;
	if (cmvn_feature_ != NULL)
		delete cmvn_feature_;
	if (base_feature_ != NULL)
		delete base_feature_;
}

int32 OnlineNnetFeaturePipeline::Dim() const {
	return final_feature_->Dim();
}

bool OnlineNnetFeaturePipeline::IsLastFrame(int32 frame) const {
	return final_feature_->IsLastFrame(frame);
}

int32 OnlineNnetFeaturePipeline::NumFramesReady() const {
	return final_feature_->NumFramesReady();
}

void OnlineNnetFeaturePipeline::GetFrame(int32 frame,
                                          VectorBase<BaseFloat> *feat) {
	return final_feature_->GetFrame(frame, feat);
}

void OnlineNnetFeaturePipeline::Reset() {
	final_feature_->Reset();
}

BaseFloat OnlineNnetFeaturePipeline::FrameShiftInSeconds() const {
	return base_feature_->FrameShiftInSeconds();
}

void OnlineNnetFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineNnetFeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
}

} // namespace kaldi



