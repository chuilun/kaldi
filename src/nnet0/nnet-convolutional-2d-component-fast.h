// nnet0/nnet-convolutional-2d-component-fast.h

// Copyright 2015-2016  Shanghai Jiao Tong University (author: Wei Deng)
// Copyright 2014-2015  Johns Hopkins University (author: Sri Harish Mallidi)
//                      Brno University of Technology (author: Karel Vesely),
//

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


#ifndef KALDI_NNET_NNET_CONVOLUTIONAL_2D_COMPONENT_FAST_H_
#define KALDI_NNET_NNET_CONVOLUTIONAL_2D_COMPONENT_FAST_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet0 {

/**
 * Convolutional2DComponent implements convolution over 2-axis (frequency and temporal)
 * (i.e. frequency axis in case we are the 1st component in NN). 
 * // We don't do convolution along temporal axis, which simplifies the 
 * // implementation (and was not helpful for Tara).
 *
 * We assume the input featrues are spliced, i.e. each frame
 * is in fact a set of stacked frames, where we can form patches
 * which span over several frequency bands and time axes.
 *
 * The convolution is done over whole axis with same filters, 
 * i.e. we don't use separate filters for different 'regions' 
 * of frequency axis.
 *
 * In order to have a fast implementations, the filters 
 * are represented in vectorized form, where each rectangular
 * filter corresponds to a row in a matrix, where all filters 
 * are stored. The features are then re-shaped to a set of matrices, 
 * where one matrix corresponds to single patch-position, 
 * where the filters get applied.
 * 
 * The type of convolution is controled by hyperparameters:
 * x_patch_dim_,y_patch_dim_     ... temporal and frequency axes sizes of the patch (e.g. (9,9) for 9x9 2D filter)
 * x_patch_step_,y_patch_step_    ... temporal and frequencey sizes of shifts in the convolution (e.g. (1,1) 2D filter with 1 step shift in both axes)
 * x_patch_stride_,y_patch_stride_  ... dimension of the feature (maps if inside convolutional layer) (e.g. (11,32) for 32-band 11 frame spliced spectrogram patch)
 * The type of convolution is controlled by hyperparameters:
 * fmap_x_len_, fmap_y_len_ ... dimension of the feature (maps if inside convolutional layer) (e.g. (11,32) for 32-band 11 frame spliced spectrogram patch)
 * filt_x_len_, filt_y_len_ ... temporal and frequency sizes of the filters (e.g. (9,9) for 9x9 2D filter)
 * filt_x_step_, filt_y_step_ ... temporal and frequency sizes of the filters (e.g. (1,1) for 2D-filter, with 1 step shift in both axes)
 * 
 *
 * Due to convolution same weights are used repeateadly, 
 * the final gradient is average of all position-specific 
 * gradients.
 *
 */
class Convolutional2DComponentFast : public UpdatableComponent {
	friend class NnetModelSync;
 public:
	Convolutional2DComponentFast(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out),
      fmap_x_len_(0), fmap_y_len_(0),
	  pad_x_len_(0), pad_y_len_(0),
      filt_x_len_(0), filt_y_len_(0),
      filt_x_step_(0), filt_y_step_(0),
      connect_fmap_(0), learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0)
  { }
  ~Convolutional2DComponentFast()
  { }

  Component* Copy() const { return new Convolutional2DComponentFast(*this); }
  ComponentType GetType() const { return kConvolutional2DComponentFast; }

  void InitData(std::istream &is) {
    // define options
    BaseFloat bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
    BaseFloat learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<FmapXLen>")    ReadBasicType(is, false, &fmap_x_len_);
      else if (token == "<FmapYLen>")    ReadBasicType(is, false, &fmap_y_len_);
      else if (token == "<PadXLen>")     ReadBasicType(is, false, &pad_x_len_);
      else if (token == "<PadYLen>")     ReadBasicType(is, false, &pad_y_len_);
      else if (token == "<FiltXLen>")    ReadBasicType(is, false, &filt_x_len_);
      else if (token == "<FiltYLen>")    ReadBasicType(is, false, &filt_y_len_);
      else if (token == "<FiltXStep>")   ReadBasicType(is, false, &filt_x_step_);
      else if (token == "<FiltYStep>")   ReadBasicType(is, false, &filt_y_step_);
      else if (token == "<ConnectFmap>") ReadBasicType(is, false, &connect_fmap_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|FmapXLen|FmapYLen|FiltXLen|FiltYLen|FiltXStep|FiltYStep|ConnectFmap|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws;  // eat-up whitespace
    }

    //
    // Sanity checks:
    //
    // input sanity checks
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    KALDI_LOG << "num_input_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
	int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
	int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
	int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
    KALDI_ASSERT((pad_fmap_x_len - filt_x_len_) % (filt_x_step_) == 0);
    KALDI_ASSERT((pad_fmap_y_len - filt_y_len_) % (filt_y_step_) == 0);

    // output sanity checks
    int32 num_output_fmaps = output_dim_ / (out_pad_fmap_x_len * out_pad_fmap_y_len);
    KALDI_ASSERT(output_dim_ % (out_pad_fmap_x_len * out_pad_fmap_y_len)  == 0);
    KALDI_LOG << "num_output_fmaps " << num_output_fmaps;
    int32 num_filters = output_dim_/(out_pad_fmap_x_len * out_pad_fmap_y_len);
    KALDI_LOG << "num_filters " << num_filters;

    //
    // Initialize parameters
    //
    Matrix<BaseFloat> mat(num_filters, num_input_fmaps*filt_x_len_*filt_y_len_);
    for (int32 r = 0; r < num_filters; r++) {
      for (int32 c = 0; c < num_input_fmaps*filt_x_len_*filt_y_len_; c++) {
        // 0-mean Gauss with given std_dev
        if (param_range == 0.0)
            mat(r, c) = param_stddev * RandGauss();
        else
            mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
      }
    }
    filters_ = mat;
    //
    Vector<BaseFloat> vec(num_filters);
    for (int32 i = 0; i < num_filters; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range;
    }
    bias_ = vec;
    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &learn_rate_coef_);
    ExpectToken(is, binary, "<BiasLearnRateCoef>");
    ReadBasicType(is, binary, &bias_learn_rate_coef_);
    // convolution hyperparameters
    ExpectToken(is, binary, "<FmapXLen>");
    ReadBasicType(is, binary, &fmap_x_len_);
    ExpectToken(is, binary, "<FmapYLen>");
    ReadBasicType(is, binary, &fmap_y_len_);
    // optional padding feature map
	 if ('P' == PeekToken(is, binary)) {
		 ExpectToken(is, binary, "<PadXLen>");
		 ReadBasicType(is, binary, &pad_x_len_);
		 ExpectToken(is, binary, "<PadYLen>");
		 ReadBasicType(is, binary, &pad_y_len_);
	 }
    ExpectToken(is, binary, "<FiltXLen>");
    ReadBasicType(is, binary, &filt_x_len_);
    ExpectToken(is, binary, "<FiltYLen>");
    ReadBasicType(is, binary, &filt_y_len_);
    ExpectToken(is, binary, "<FiltXStep>");
    ReadBasicType(is, binary, &filt_x_step_);
    ExpectToken(is, binary, "<FiltYStep>");
    ReadBasicType(is, binary, &filt_y_step_);
    ExpectToken(is, binary, "<ConnectFmap>");
    ReadBasicType(is, binary, &connect_fmap_);

    // trainable parameters
    ExpectToken(is, binary, "<Filters>");
    filters_.Read(is, binary);
    ExpectToken(is, binary, "<Bias>");
    bias_.Read(is, binary);


    //
    // Sanity checks:
    //
    // input sanity checks
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
	int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
	int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
	int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
	int32 out_pad_fmap_size = out_pad_fmap_x_len*out_pad_fmap_y_len;
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);

    //int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    //KALDI_LOG << "num_input_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    KALDI_ASSERT((pad_fmap_x_len - filt_x_len_) % (filt_x_step_) == 0);
    KALDI_ASSERT((pad_fmap_y_len - filt_y_len_) % (filt_y_step_) == 0);

    // output sanity checks
    KALDI_ASSERT(output_dim_ % (out_pad_fmap_x_len * out_pad_fmap_y_len)  == 0);

    // init grad size
    filters_grad_.Resize(filters_.NumRows(), filters_.NumCols(), kSetZero);
    bias_grad_.Resize(filters_.NumRows(), kSetZero);
    filters_grad_patches_.Resize(out_pad_fmap_size*filters_.NumRows(), filters_.NumCols(), kSetZero);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    // convolution hyperparameters
    WriteToken(os, binary, "<FmapXLen>");
    WriteBasicType(os, binary, fmap_x_len_);
    WriteToken(os, binary, "<FmapYLen>");
    WriteBasicType(os, binary, fmap_y_len_);
    if (pad_x_len_ > 0 || pad_y_len_ > 0) {
    	WriteToken(os, binary, "<PadXLen>");
    	WriteBasicType(os, binary, pad_x_len_);
    	WriteToken(os, binary, "<PadYLen>");
    	WriteBasicType(os, binary, pad_y_len_);
    }
    WriteToken(os, binary, "<FiltXLen>");
    WriteBasicType(os, binary, filt_x_len_);
    WriteToken(os, binary, "<FiltYLen>");
    WriteBasicType(os, binary, filt_y_len_);
    WriteToken(os, binary, "<FiltXStep>");
    WriteBasicType(os, binary, filt_x_step_);
    WriteToken(os, binary, "<FiltYStep>");
    WriteBasicType(os, binary, filt_y_step_);
    WriteToken(os, binary, "<ConnectFmap>");
    WriteBasicType(os, binary, connect_fmap_);

    // trainable parameters
    WriteToken(os, binary, "<Filters>");
    filters_.Write(os, binary);
    WriteToken(os, binary, "<Bias>");
    bias_.Write(os, binary);
  }

  int32 NumParams() const {
    return filters_.NumRows()*filters_.NumCols() + bias_.Dim();
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 filters_num_elem = filters_.NumRows() * filters_.NumCols();
    wei_copy->Range(0, filters_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(filters_));
    wei_copy->Range(filters_num_elem, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
  }

  std::string Info() const {
    return std::string("\n  filters") + MomentStatistics(filters_) +
           "\n  bias" + MomentStatistics(bias_);
  }
  std::string InfoGradient() const {
    return std::string("\n  filters_grad") + MomentStatistics(filters_grad_) +
           ", lr-coef " + ToString(learn_rate_coef_) +
           "\n  bias_grad" + MomentStatistics(bias_grad_) +
           ", lr-coef " + ToString(bias_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    // int32 inp_fmap_size = fmap_x_len_ * fmap_y_len_;
    // int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);

    // for padding
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
    int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
    int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
    int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
    int32 out_pad_fmap_size = out_pad_fmap_x_len*out_pad_fmap_y_len;
    int32 pad_input_dim = num_input_fmaps * pad_fmap_x_len * pad_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_pad_fmap_x_len * out_pad_fmap_y_len);

    // this is total num_filters,
    // so each input_fmap has size num_filters/num_input_fmaps
    int32 num_filters = filters_.NumRows();
    KALDI_ASSERT(num_filters == num_output_fmaps);
    // int32 filter_size = filt_x_len_*filt_y_len_;
    int32 num_frames = in.NumRows();

    // padding the feature map
    if (input_pad_.NumRows() != in.NumRows() && (pad_x_len_>0 || pad_y_len_>0)) {
    	input_pad_.Resize(in.NumRows(), pad_input_dim, kSetZero);
    }
    const CuMatrixBase<BaseFloat> &input = (pad_x_len_>0 || pad_y_len_>0) ? input_pad_ : in;

    if (pad_x_len_>0 || pad_y_len_>0)
        input_pad_.PadFeatureMap(in, num_input_fmaps, fmap_x_len_, fmap_y_len_, pad_x_len_, pad_y_len_, 0, connect_fmap_);

    // we will need the buffers
    input_patches_.Resize(out_pad_fmap_size*num_frames, filters_.NumCols(), kUndefined);
    forward_output_.Resize(out_pad_fmap_size*num_frames, filters_.NumRows(), kUndefined);

    input_patches_.ConvolutionForwardExpandWorkspace(input, num_input_fmaps, pad_fmap_x_len, pad_fmap_y_len,
    		filt_x_len_, filt_y_len_, filt_x_step_, filt_y_step_, connect_fmap_);

    forward_output_.AddMatMat(1.0, input_patches_, kNoTrans, filters_, kTrans, 0.0);
    forward_output_.AddVecToRows(1.0, bias_, 1.0);

    // we will need the buffers
    if (vectorized_forward_patches_.size() != out_pad_fmap_size) {
    	vectorized_forward_patches_.resize(out_pad_fmap_size);
    }

	for (int32 p = 0; p < out_pad_fmap_size; p++) {
		CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(forward_output_.RowRange(p*num_frames, num_frames));
		vectorized_forward_patches_[p] = tgt;
	}

    out->CopyColMats(vectorized_forward_patches_);

	for (int32 p = 0; p < out_pad_fmap_size; p++)
		delete vectorized_forward_patches_[p];
  }


  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);

    // int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len * out_fmap_y_len;
    // int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);

    // for padding
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
    int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
    int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
    int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
    int32 out_pad_fmap_size = out_pad_fmap_x_len*out_pad_fmap_y_len;
    int32 pad_input_dim = num_input_fmaps * pad_fmap_x_len * pad_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_pad_fmap_x_len * out_pad_fmap_y_len);

    // this is total num_filters,
    // so each input_fmap has num_filters/num_input_fmaps
    int32 num_filters = filters_.NumRows();
    KALDI_ASSERT(num_filters == num_output_fmaps);
    // int32 filter_size = filt_x_len_*filt_y_len_;
    int32 num_frames = in.NumRows();

    // padding the feature map
	if (indiff_pad_.NumRows() != in_diff->NumRows() && (pad_x_len_>0 || pad_y_len_>0))
		indiff_pad_.Resize(in_diff->NumRows(), pad_input_dim, kSetZero);
	CuMatrixBase<BaseFloat> *indiff = (pad_x_len_>0 || pad_y_len_>0) ? &indiff_pad_ : in_diff;

    // compute in_diff_maps once
    if (in_diff_summands_.Dim() == 0)
    {
    	int32 out_fmap_cnt = 0;
    	int32 index = 0;
    	std::vector<std::vector<Int32Pair> > map(pad_input_dim);
    	std::vector<int32> mapsize(pad_input_dim, 0);

    	for (int32 m = 0; m < pad_fmap_x_len-filt_x_len_+1; m = m+filt_x_step_)
    	{
            for (int32 n = 0; n < pad_fmap_y_len-filt_y_len_+1; n = n+filt_y_step_)
            {
              int32 st = 0;
              if (connect_fmap_ == 1) {
                st = (m * pad_fmap_y_len + n) * num_input_fmaps;
              } else {
                st = m * pad_fmap_y_len * num_input_fmaps + n;
              }
              for (int32 i = 0; i < filt_x_len_; i++) {
                for (int32 j = 0; j < filt_y_len_*num_input_fmaps; j++) {
                  int32 c = 0;
                  if (connect_fmap_ == 1) {
                    c = st + i * (num_input_fmaps * pad_fmap_y_len) + j;
                  } else {
                    c = st + i * (num_input_fmaps * pad_fmap_y_len)
                           + (j / num_input_fmaps)
                           + (j % num_input_fmaps) * pad_fmap_y_len;
                  }

                  index = i * (num_input_fmaps * filt_y_len_) + j;
                  Int32Pair tmp = {out_fmap_cnt, index};
                  map[c].push_back(tmp);
                  mapsize[c]++;

                }
              }
              out_fmap_cnt++;
            }
          }

			CuArray<Int32Pair> *tmp;
    		std::vector<Int32Pair*> pmap;
    		Vector<BaseFloat> summands(pad_input_dim);
    		BaseFloat *summands_data = summands.Data();

    		for (int i = 0; i < map.size(); i++)
    		{
    			tmp = new CuArray<Int32Pair>(map[i]);
    			pmap.push_back(tmp->Data());
    			summands_data[i] = mapsize[i];
    		}

    		in_diff_map.Resize(pad_input_dim);
    		in_diff_mapsize_.Resize(pad_input_dim);
    		in_diff_summands_.Resize(pad_input_dim);

    		in_diff_map.CopyFromVec(pmap);
    		in_diff_mapsize_.CopyFromVec(mapsize);
    		in_diff_summands_.CopyFromVec(summands);

    		in_diff_summands_.InvertElements();
    }


    	// we will need the buffers
       if (vectorized_diff_patches_.size() != out_pad_fmap_size)
		vectorized_diff_patches_.resize(out_pad_fmap_size);

	   for (int32 p = 0; p < out_pad_fmap_size; p++) {
			CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(out_diff.ColRange(p*num_filters, num_filters));
			vectorized_diff_patches_[p] = tgt;
	   }

       diff_output_.Resize(num_frames*out_pad_fmap_size, num_filters, kUndefined);
       diff_patches_.Resize(num_frames*out_pad_fmap_size, filters_.NumCols(), kUndefined);

       diff_output_.CopyRowMats(vectorized_diff_patches_);

       diff_patches_.AddMatMat(1.0, diff_output_, kNoTrans, filters_, kNoTrans, 0.0);


       indiff->ConvolutionBackwardShrinkWorkspace(diff_patches_, in_diff_map, in_diff_mapsize_);

       // compensate for summands
       indiff->MulColsVec(in_diff_summands_);

       if (pad_x_len_>0 || pad_y_len_>0)
    	   in_diff->PadFeatureMap(indiff_pad_, num_input_fmaps, fmap_x_len_, fmap_y_len_, pad_x_len_, pad_y_len_, 1, connect_fmap_);
  }

  void ResetGradient()
  {
      filters_grad_patches_.SetZero();
      bias_grad_.SetZero();
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // useful dims
    // int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);

    // for padding
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
    int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
    int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
    int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
    int32 out_pad_fmap_size = out_pad_fmap_x_len*out_pad_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_pad_fmap_x_len * out_pad_fmap_y_len);

    int32 num_filters = filters_.NumRows();  // this is total num_filters, so each input_fmap has num_filters/num_input_fmaps
    KALDI_ASSERT(num_filters == num_output_fmaps);
    int32 num_frames = input.NumRows();

    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    //const BaseFloat l1 = opts_.l1_penalty;

    //
    // calculate the gradient
    //
    //filters_grad_.Resize(filters_.NumRows(), filters_.NumCols(), kSetZero);
    //bias_grad_.Resize(filters_.NumRows(), kSetZero);
    //filters_grad_patches_.Resize(out_fmap_size*filters_.NumRows(), filters_.NumCols(), kSetZero);


    if (vectorized_input_patches_.size() != out_pad_fmap_size)
	vectorized_input_patches_.resize(out_pad_fmap_size);

    for (int32 p = 0; p < out_pad_fmap_size; p++) {
             CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(input_patches_.RowRange(p*num_frames, num_frames));
             vectorized_input_patches_[p] = tgt;
    }   

    if (vectorized_grad_patches_.size() != out_pad_fmap_size)
	vectorized_grad_patches_.resize(out_pad_fmap_size);

    for (int32 p = 0; p < out_pad_fmap_size; p++) {
   		CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(filters_grad_patches_.RowRange(p*num_filters, num_filters));
    		vectorized_grad_patches_[p] = tgt;
    }

    // compute gradient (incl. momentum)
    AddMatMatBatched(static_cast<BaseFloat>(1.0f), vectorized_grad_patches_, vectorized_diff_patches_, kTrans, vectorized_input_patches_, kNoTrans, 
	static_cast<BaseFloat>(mmt));
    filters_grad_.SumMats(vectorized_grad_patches_);
    bias_grad_.AddRowSumMat(1.0, diff_output_, mmt);

    // scale
    filters_grad_.Scale(1.0/out_pad_fmap_size);
    bias_grad_.Scale(1.0/out_pad_fmap_size);

    // l2 regularization
	if (l2 != 0.0) {
	  filters_.AddMat(-lr*l2*num_frames, filters_);
	}

    /* NOT NOW:

    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&filters_, &filters_grad_, lr*l1*num_frames, lr);
    }
    */

    for (int32 p = 0; p < out_pad_fmap_size; p++)
    {
		delete vectorized_input_patches_[p];
		delete vectorized_grad_patches_[p];
		delete vectorized_diff_patches_[p];
    }
 }

 void UpdateGradient(){
	
    const BaseFloat lr = opts_.learn_rate;
    //
    // update
    //
    filters_.AddMat(-lr*learn_rate_coef_, filters_grad_);
    bias_.AddVec(-lr*bias_learn_rate_coef_, bias_grad_);
    //
}


  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // useful dims
    // int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);

    // for padding
    int32 pad_fmap_x_len = fmap_x_len_ + 2*pad_x_len_;
    int32 pad_fmap_y_len = fmap_y_len_ + 2*pad_y_len_;
    int32 out_pad_fmap_x_len = (pad_fmap_x_len - filt_x_len_)/filt_x_step_ + 1;
    int32 out_pad_fmap_y_len = (pad_fmap_y_len - filt_y_len_)/filt_y_step_ + 1;
    int32 out_pad_fmap_size = out_pad_fmap_x_len*out_pad_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_pad_fmap_x_len * out_pad_fmap_y_len);

    int32 num_filters = filters_.NumRows();  // this is total num_filters, so each input_fmap has num_filters/num_input_fmaps
    KALDI_ASSERT(num_filters == num_output_fmaps);
    int32 num_frames = input.NumRows();

    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;
    //const BaseFloat mmt = 0.0;
    //const BaseFloat l2 = opts_.l2_penalty;
    //const BaseFloat l1 = opts_.l1_penalty;
    /* NOT NOW:
    */


    //
    // calculate the gradient
    //
    //filters_grad_.Resize(filters_.NumRows(), filters_.NumCols(), kSetZero);
    //bias_grad_.Resize(filters_.NumRows(), kSetZero);
    //filters_grad_patches_.Resize(out_fmap_size*filters_.NumRows(), filters_.NumCols(), kSetZero);


    if (vectorized_input_patches_.size() != out_pad_fmap_size)
	vectorized_input_patches_.resize(out_pad_fmap_size);

    for (int32 p = 0; p < out_pad_fmap_size; p++) {
             CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(input_patches_.RowRange(p*num_frames, num_frames));
             vectorized_input_patches_[p] = tgt;
    }   

    if (vectorized_grad_patches_.size() != out_pad_fmap_size)
	vectorized_grad_patches_.resize(out_pad_fmap_size);

    for (int32 p = 0; p < out_pad_fmap_size; p++) {
   		CuSubMatrix<BaseFloat> *tgt = new CuSubMatrix<BaseFloat>(filters_grad_patches_.RowRange(p*num_filters, num_filters));
    		vectorized_grad_patches_[p] = tgt;
    }

    // compute gradient (incl. momentum)
    AddMatMatBatched(static_cast<BaseFloat>(1.0f), vectorized_grad_patches_, vectorized_diff_patches_, kTrans, vectorized_input_patches_, kNoTrans, static_cast<BaseFloat>(mmt));
    filters_grad_.SumMats(vectorized_grad_patches_);
    bias_grad_.AddRowSumMat(1.0, diff_output_, mmt);

    // scale
    filters_grad_.Scale(1.0/out_pad_fmap_size);
    bias_grad_.Scale(1.0/out_pad_fmap_size);


    /* NOT NOW:
    // l2 regularization
    if (l2 != 0.0) {
      filters_.AddMat(-lr*l2*num_frames, filters_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&filters_, &filters_grad_, lr*l1*num_frames, lr);
    }
    */

    //
    // update
    //
    filters_.AddMat(-lr*learn_rate_coef_, filters_grad_);
    bias_.AddVec(-lr*bias_learn_rate_coef_, bias_grad_);
    //

    //delete[] &vectorized_input_patches_.front();

    for (int32 p = 0; p < out_pad_fmap_size; p++)
    {
		delete vectorized_input_patches_[p];
		delete vectorized_grad_patches_[p];
		delete vectorized_diff_patches_[p];
    }
  }

  int32 GetDim() const {
	  return filters_.SizeInBytes()/sizeof(BaseFloat) + bias_.Dim();
  }


  int WeightCopy(void *host, int direction, int copykind) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        Timer tim;

        int32 dst_pitch, src_pitch, width,  size;
        int pos = 0;
        void *src, *dst;
        MatrixDim dim;
        cudaMemcpyKind kind;
        switch(copykind)
        {
            case 0:
                kind = cudaMemcpyHostToHost;
                break;
            case 1:
                kind = cudaMemcpyHostToDevice;
                break;
            case 2:
                kind = cudaMemcpyDeviceToHost;
                break;
            case 3:
                kind = cudaMemcpyDeviceToDevice;
                break;
            default:
                KALDI_ERR << "Default based unified virtual address space";
                break;
        }

		dim = filters_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)filters_.Data());
		src = (void*) (direction==0 ? (char *)filters_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += filters_.SizeInBytes();

		size = bias_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)bias_.Data());
		src = (void*) (direction==0 ? (char *)bias_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

  	  CU_SAFE_CALL(cudaGetLastError());

  	  CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());

  	  return pos;
  }else
#endif
  	{
  		// not implemented for CPU yet
  		return 0;
  	}
  }

 private:
  int32 fmap_x_len_, fmap_y_len_,  ///< feature maps dimensions (for input x_ is usually splice and y_ is num of fbanks) shift for 2nd dim of a patch (i.e. frame length before splicing)
	pad_x_len_, pad_y_len_,	 ///< 2D padding dimensions, x_ temporal, y_ spectral
  	filt_x_len_, filt_y_len_,  ///< 2D filter dimensions, x_ temporal, y_ spectral
    filt_x_step_, filt_y_step_,  ///< 2D shifts along temporal and spectral
    connect_fmap_;  ///< if connect_fmap_ = 1, then each fmap has num_filt

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;

  CuMatrix<BaseFloat> input_pad_;  ///< padded input feature maps
  CuMatrix<BaseFloat> indiff_pad_;  ///< padded input feature maps

  CuMatrix<BaseFloat> filters_;  ///< row = vectorized rectangular filter
  CuVector<BaseFloat> bias_;  ///< bias for each filter

  CuMatrix<BaseFloat> filters_grad_;  ///< gradient of filters
  CuVector<BaseFloat> bias_grad_;  ///< gradient of biases
  std::vector<CuSubMatrix<BaseFloat>* > vectorized_grad_patches_;
  CuMatrix<BaseFloat> filters_grad_patches_;

  /** Buffer of reshaped inputs:
   *  1row = vectorized rectangular feature patch,
   *  1col = dim over speech frames,
   *  std::vector-dim = patch-position
   */
  std::vector<CuSubMatrix<BaseFloat>* > vectorized_input_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > vectorized_forward_patches_;
  CuMatrix<BaseFloat> input_patches_;
  CuMatrix<BaseFloat> forward_output_;

  /** Buffer for backpropagation:
   *  derivatives in the domain of 'vectorized_feature_patches_',
   *  1row = vectorized rectangular feature patch,
   *  1col = dim over speech frames,
   *  std::vector-dim = patch-position
   */
  std::vector<CuSubMatrix<BaseFloat>* > vectorized_diff_patches_;
  CuMatrix<BaseFloat> diff_patches_;
  CuMatrix<BaseFloat> diff_output_;

  /// Auxiliary vector for compensating #summands when backpropagating
  CuVector<BaseFloat> in_diff_summands_;

  CuArray<Int32Pair* > in_diff_map;
  CuArray<int32>	in_diff_mapsize_;
};

}  // namespace nnet0
}  // namespace kaldi

#endif
