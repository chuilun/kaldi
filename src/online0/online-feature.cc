// online0/online-feature.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer

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

#include "online0/online-feature.h"

namespace kaldi {

template<class C>
void OnlineStreamGenericBaseFeature<C>::GetFrame(int32 frame,
                                           VectorBase<BaseFloat> *feat) {
  // 'at' does size checking.
  feat->CopyFromVec(*(features_.at(frame)));
};

template<class C>
OnlineStreamGenericBaseFeature<C>::OnlineStreamGenericBaseFeature(
    const typename C::Options &opts):
    computer_(opts), window_function_(computer_.GetFrameOptions()),
    input_finished_(false), waveform_offset_(0) { }

template<class C>
void OnlineStreamGenericBaseFeature<C>::AcceptWaveform(BaseFloat sampling_rate,
                                                 const VectorBase<BaseFloat> &waveform) {
  BaseFloat expected_sampling_rate = computer_.GetFrameOptions().samp_freq;
  if (sampling_rate != expected_sampling_rate)
    KALDI_ERR << "Sampling frequency mismatch, expected "
              << expected_sampling_rate << ", got " << sampling_rate;
  if (waveform.Dim() == 0)
    return;  // Nothing to do.
  if (input_finished_)
    KALDI_ERR << "AcceptWaveform called after InputFinished() was called.";
  // append 'waveform' to 'waveform_remainder_.'
  Vector<BaseFloat> appended_wave(waveform_remainder_.Dim() + waveform.Dim());
  if (waveform_remainder_.Dim() != 0)
    appended_wave.Range(0, waveform_remainder_.Dim()).CopyFromVec(
        waveform_remainder_);
  appended_wave.Range(waveform_remainder_.Dim(), waveform.Dim()).CopyFromVec(
      waveform);
  waveform_remainder_.Swap(&appended_wave);
  ComputeFeatures();
}

template<class C>
void OnlineStreamGenericBaseFeature<C>::ComputeFeatures() {
  const FrameExtractionOptions &frame_opts = computer_.GetFrameOptions();
  int64 num_samples_total = waveform_offset_ + waveform_remainder_.Dim();
  int32 num_frames_old = features_.size(),
      num_frames_new = NumFrames(num_samples_total, frame_opts,
                                 input_finished_);
  KALDI_ASSERT(num_frames_new >= num_frames_old);
  features_.resize(num_frames_new, NULL);

  Vector<BaseFloat> window;
  bool need_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 frame = num_frames_old; frame < num_frames_new; frame++) {
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(waveform_offset_, waveform_remainder_, frame,
                  frame_opts, window_function_, &window,
                  need_raw_log_energy ? &raw_log_energy : NULL);
    Vector<BaseFloat> *this_feature = new Vector<BaseFloat>(computer_.Dim(),
                                                            kUndefined);
    // note: this online feature-extraction code does not support VTLN.
    BaseFloat vtln_warp = 1.0;
    computer_.Compute(raw_log_energy, vtln_warp, &window, this_feature);
    features_[frame] = this_feature;
  }
  // OK, we will now discard any portion of the signal that will not be
  // necessary to compute frames in the future.
  int64 first_sample_of_next_frame = FirstSampleOfFrame(num_frames_new,
                                                        frame_opts);
  int32 samples_to_discard = first_sample_of_next_frame - waveform_offset_;
  if (samples_to_discard > 0) {
    // discard the leftmost part of the waveform that we no longer need.
    int32 new_num_samples = waveform_remainder_.Dim() - samples_to_discard;
    if (new_num_samples <= 0) {
      // odd, but we'll try to handle it.
      waveform_offset_ += waveform_remainder_.Dim();
      waveform_remainder_.Resize(0);
    } else {
      Vector<BaseFloat> new_remainder(new_num_samples);
      new_remainder.CopyFromVec(waveform_remainder_.Range(samples_to_discard,
                                                          new_num_samples));
      waveform_offset_ += samples_to_discard;
      waveform_remainder_.Swap(&new_remainder);
    }
  }
}

// instantiate the templates defined here for MFCC, PLP and filterbank classes.
template class OnlineStreamGenericBaseFeature<MfccComputer>;
template class OnlineStreamGenericBaseFeature<PlpComputer>;
template class OnlineStreamGenericBaseFeature<FbankComputer>;


int32 OnlineStreamDeltaFeature::Dim() const {
  int32 src_dim = src_->Dim();
  return src_dim * (1 + opts_.order);
}

int32 OnlineStreamDeltaFeature::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady(),
      context = opts_.order * opts_.window;
  // "context" is the number of frames on the left or (more relevant
  // here) right which we need in order to produce the output.
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - context);
}

void OnlineStreamDeltaFeature::GetFrame(int32 frame,
                                      VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
  KALDI_ASSERT(feat->Dim() == Dim());
  // We'll produce a temporary matrix containing the features we want to
  // compute deltas on, but truncated to the necessary context.
  int32 context = opts_.order * opts_.window;
  int32 left_frame = frame - context,
      right_frame = frame + context,
      src_frames_ready = src_->NumFramesReady();
  if (left_frame < 0) left_frame = 0;
  if (right_frame >= src_frames_ready)
    right_frame = src_frames_ready - 1;
  KALDI_ASSERT(right_frame >= left_frame);
  int32 temp_num_frames = right_frame + 1 - left_frame,
      src_dim = src_->Dim();
  Matrix<BaseFloat> temp_src(temp_num_frames, src_dim);
  for (int32 t = left_frame; t <= right_frame; t++) {
    SubVector<BaseFloat> temp_row(temp_src, t - left_frame);
    src_->GetFrame(t, &temp_row);
  }
  int32 temp_t = frame - left_frame;  // temp_t is the offset of frame "frame"
                                      // within temp_src
  delta_features_.Process(temp_src, temp_t, feat);
}


OnlineStreamDeltaFeature::OnlineStreamDeltaFeature(const DeltaFeaturesOptions &opts,
                                       OnlineStreamFeatureInterface *src):
    src_(src), opts_(opts), delta_features_(opts) { }


/**
   This class does an online version of the cepstral mean and [optionally]
   variance, but note that this is not equivalent to the offline version.  This
   is necessarily so, as the offline computation involves looking into the
   future.  If you plan to use features normalized with this type of CMVN then
   you need to train in a `matched' way, i.e. with the same type of features.
   We normally only do so in the "online" nnet-based decoding, e.g.nnet0
*/

OnlineStreamCmvnFeature::OnlineStreamCmvnFeature(const OnlineStreamCmvnOptions &opts,
                   OnlineStreamFeatureInterface *src):
                		   opts_(opts), src_(src)
{
	KALDI_ASSERT(opts_.min_window >= 0);
	if (opts_.cmn_window >= 0)
		KALDI_ASSERT(opts_.cmn_window >= opts_.min_window);

	sum_.Resize(src_->Dim());
	sumsq_.Resize(src_->Dim());
	features_.resize(0);
}

int32 OnlineStreamCmvnFeature::NumFramesReady() const
{
	int src_frames_ready = src_->NumFramesReady();
	bool isfinished = src_->IsLastFrame(src_frames_ready-1);
	if (!isfinished && src_frames_ready < opts_.min_window)
		return 0;
    
	return src_frames_ready;
}

void OnlineStreamCmvnFeature::ComputeCmvnInternal()
{
	int src_frames_ready = src_->NumFramesReady();
	int curt_frames_ready = features_.size();
    bool isfinished = src_->IsLastFrame(src_frames_ready-1);

    Vector<BaseFloat> *this_feature = NULL;

	if (src_frames_ready >= opts_.min_window || isfinished) {
		for (int i = curt_frames_ready; i < src_frames_ready; i++) {
			 this_feature = new Vector<BaseFloat>(src_->Dim(), kUndefined);
			if (opts_.cmn_window >= 0 && i >= opts_.cmn_window) {
				src_->GetFrame(i-opts_.cmn_window, this_feature);
				sum_.AddVec(-1.0, *this_feature);
				if (opts_.normalize_variance) {
					sumsq_.AddVec2(-1.0, *this_feature);
				}
			}

			src_->GetFrame(i, this_feature);
			sum_.AddVec(1.0, *this_feature);
			sumsq_.AddVec2(1.0, *this_feature);
			features_.push_back(this_feature);

			// normalize min_window
			if (i == opts_.min_window-1 || (isfinished && src_frames_ready <= opts_.min_window && i == src_frames_ready-1)) {
				int num_frames = (isfinished && src_frames_ready <= opts_.min_window) ? src_frames_ready : opts_.min_window;
				for (int j = 0; j < num_frames; j++) {
					if (opts_.normalize_mean)
						features_[j]->AddVec(-1.0/num_frames, sum_);
					if (opts_.normalize_variance) {
						Vector<double> variance(sumsq_);
						variance.Scale(1.0/num_frames);
						variance.AddVec2(-1.0/(num_frames * num_frames), sum_);
						int32 num_floored = variance.ApplyFloor(1.0e-10);
						if (num_floored > 0) {
						  KALDI_WARN << "Flooring variance When normalizing variance, floored " << num_floored
									 << " elements; num-frames was " << num_frames;
						}
						variance.ApplyPow(-0.5); // get inverse standard deviation.
						features_[j]->MulElements(variance);
					}
				}
			}

			// normalize exceed min_window
			if (i >= opts_.min_window) {
				int num_frames = (opts_.cmn_window < 0 || i < opts_.cmn_window) ? i+1 : opts_.cmn_window;
				if (opts_.normalize_mean)
					features_[i]->AddVec(-1.0/num_frames, sum_);
				if (opts_.normalize_variance) {
					Vector<double> variance(sumsq_);
					variance.Scale(1.0/num_frames);
					variance.AddVec2(-1.0/(num_frames * num_frames), sum_);
					int32 num_floored = variance.ApplyFloor(1.0e-10);
					if (num_floored > 0) {
					  KALDI_WARN << "Flooring variance When normalizing variance, floored " << num_floored
								 << " elements; num-frames was " << num_frames;
					}
					variance.ApplyPow(-0.5); // get inverse standard deviation.
					features_[i]->MulElements(variance);
				}
			}
		} // end for
	}

}

void OnlineStreamCmvnFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat)
{
    ComputeCmvnInternal();

	// 'at' does size checking.
	feat->CopyFromVec(*(features_.at(frame)));
}


/**
 * splice feature
 */
OnlineStreamSpliceFeature::OnlineStreamSpliceFeature(const OnlineStreamSpliceOptions &opts,
                     OnlineStreamFeatureInterface *src): src_(src) {
	left_context_ = opts.custom_splice ? opts.left_context : opts.context;
	right_context_ = opts.custom_splice ? opts.right_context : opts.context;
}

int32 OnlineStreamSpliceFeature::NumFramesReady() const {
  int32 num_frames = src_->NumFramesReady();
  if (num_frames > 0 && src_->IsLastFrame(num_frames-1))
    return num_frames;
  else
    return std::max<int32>(0, num_frames - right_context_);
}

void OnlineStreamSpliceFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(left_context_ >= 0 && right_context_ >= 0);
  KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
  int32 dim_in = src_->Dim();
  KALDI_ASSERT(feat->Dim() == dim_in * (1 + left_context_ + right_context_));
  int32 T = src_->NumFramesReady();
  for (int32 t2 = frame - left_context_; t2 <= frame + right_context_; t2++) {
    int32 t2_limited = t2;
    if (t2_limited < 0) t2_limited = 0;
    if (t2_limited >= T) t2_limited = T - 1;
    int32 n = t2 - (frame - left_context_);  // 0 for left-most frame,
                                             // increases to the right.
    SubVector<BaseFloat> part(*feat, n * dim_in, dim_in);
    src_->GetFrame(t2_limited, &part);
  }
}


}  // namespace kaldi
