// nnet0/nnet-time-delay-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)
// Copyright 2015-2016  Shanghai Jiao Tong University (author: Wei Deng)

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


#ifndef KALDI_NNET_NNET_TIME_DELAY_TRANSFORM_H_
#define KALDI_NNET_NNET_TIME_DELAY_TRANSFORM_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class TimeDelayTransform : public UpdatableComponent {

	friend class NnetModelSync;
	friend class lm::LmModelSync;

 public:
	TimeDelayTransform(int32 dim_in_, int32 dim_out_)
    : UpdatableComponent(dim_in_, dim_out_),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0)
  { }
  ~TimeDelayTransform()
  { }

  Component* Copy() const { return new TimeDelayTransform(*this); }
  ComponentType GetType() const { return kTimeDelayTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    int32 index = 0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm);
      else if (token == "<NumInputContext>") ReadBasicType(is,false,&num_input_context_);
      else if (token == "<NumOutputContext>") ReadBasicType(is, false, &num_output_context_);
      else if (token == "<NumIndexes>") {
		  ReadBasicType(is, false, &num_indexes_);
		  for(int i = 0; i < num_output_context_; i++)
		  {
			  ExpectToken(is, false, "<Indexes>");
			  std::vector<int32> indexes;
			  for(int j = 0; j< num_indexes_; j++)
			  {
				  ReadBasicType(is, false, &index);
				  indexes.push_back(index);
			  }
			  input_context_indexes_.push_back(indexes);
		  }
      }
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    KALDI_ASSERT(input_dim_ % num_input_context_ == 0);
    KALDI_ASSERT(output_dim_ % num_output_context_ == 0);

    	int dim_in_ = input_dim_ / num_input_context_;
    	int dim_out_ = output_dim_ / num_output_context_;

    Matrix<BaseFloat> mat(dim_out_, dim_in_*num_indexes_);
    for (int32 r=0; r<dim_out_; r++) {
      for (int32 c=0; c<dim_in_*num_indexes_; c++) {
        if (param_range == 0.0)
        	mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
	else
		mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
      }
    }
    linearity_ = mat;

    //
    Vector<BaseFloat> vec(dim_out_);
    for (int32 i=0; i<dim_out_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;
    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);

		ExpectToken(is,binary, "<NumInputContext>");
		ReadBasicType(is, binary, &num_input_context_);
		ExpectToken(is,binary, "<NumOutputContext>");
		ReadBasicType(is, binary, &num_output_context_);
		ExpectToken(is,binary, "<NumIndexes>");
		ReadBasicType(is, binary, &num_indexes_);
		input_context_indexes_.resize(num_output_context_);

		for(int i = 0; i < num_output_context_; i++) {
			input_context_indexes_[i].resize(num_indexes_);
			for(int j = 0; j < num_indexes_; j++)
				ReadBasicType(is, binary, &input_context_indexes_[i][j]);
		}
    }
    // weights
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    dim_in_ = input_dim_ / num_input_context_;
    dim_out_ = output_dim_ / num_output_context_;
    linearity_corr_.Resize(dim_out_, dim_in_ * num_indexes_);
    bias_corr_.Resize(dim_out_);

    KALDI_ASSERT(linearity_.NumRows() == dim_out_);
    KALDI_ASSERT(linearity_.NumCols() == dim_in_ * num_indexes_);
    KALDI_ASSERT(bias_.Dim() == dim_out_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);

    WriteToken(os,binary, "<NumInputContext>");
	WriteBasicType(os, binary, num_input_context_);
	WriteToken(os,binary, "<NumOutputContext>");
	WriteBasicType(os, binary, num_output_context_);
	WriteToken(os,binary, "<NumIndexes>");
	WriteBasicType(os, binary, num_indexes_);
	for(int i = 0; i < num_output_context_; i++) {
		for(int j = 0; j < num_indexes_; j++)
			WriteBasicType(os, binary, input_context_indexes_[i][j]);
	}

    // weights
	linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  int32 NumParams() const { return linearity_.NumRows()*linearity_.NumCols() + bias_.Dim(); }
  
  int32 GetDim() const { return linearity_.SizeInBytes()/sizeof(BaseFloat) + bias_.Dim(); }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 linearity_num_elem = linearity_.NumRows() * linearity_.NumCols(); 
    wei_copy->Range(0,linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_));
    wei_copy->Range(linearity_num_elem, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
  }
  
  std::string Info() const {
    return std::string("\n  linearity") + MomentStatistics(linearity_) +
           "\n  bias" + MomentStatistics(bias_);
  }
  std::string InfoGradient() const {
    return std::string("\n  linearity_grad") + MomentStatistics(linearity_corr_) + 
           ", lr-coef " + ToString(learn_rate_coef_) +
           ", max-norm " + ToString(max_norm_) +
           "\n  bias_grad" + MomentStatistics(bias_corr_) + 
           ", lr-coef " + ToString(bias_learn_rate_coef_);
           
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	  int32 num_frames = in.NumRows();

	  // we will need the buffers
	  input_patches_.Resize(num_frames*num_output_context_, dim_in_*num_indexes_, kUndefined);
	  forward_output_.Resize(num_frames*num_output_context_, dim_out_, kUndefined);

	  for(int i = 0; i < num_output_context_; i++) {
		  for(int j = 0; j < num_indexes_; j++)
			  input_patches_.RowRange(i*num_frames, num_frames).ColRange(j*dim_in_, dim_in_).CopyFromMat(in.ColRange(input_context_indexes_[i][j]*dim_in_, dim_in_));
	  }

	  // multiply by weights^t
	  forward_output_.AddMatMat(1.0, input_patches_, kNoTrans, linearity_, kTrans, 0.0);
	  forward_output_.AddVecToRows(1.0, bias_,1.0);

	  // rearrange the output buffer
	  for(int i = 0; i < num_output_context_; i++)
		  out->ColRange(i*dim_out_, dim_out_).CopyFromMat(forward_output_.RowRange(i*num_frames, num_frames));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
	  int32 num_frames = in.NumRows();
	  in_diff->SetZero();

	  // we will need the buffers
	  diff_output_.Resize(num_frames*num_output_context_, dim_in_*num_indexes_, kUndefined);
	  diff_patches_.Resize(num_frames*num_output_context_, dim_out_, kUndefined);

	  for(int i = 0; i < num_output_context_; i++)
		  diff_patches_.RowRange(i*num_frames, num_frames).CopyFromMat(out_diff.ColRange(i*dim_out_, dim_out_));

	  // multiply error derivative by weights
	  diff_output_.AddMatMat(1.0, diff_patches_, kNoTrans, linearity_, kNoTrans, 0.0);

	  for(int i = 0; i < num_output_context_; i++)
	  	  for(int j = 0; j < num_indexes_; j++)
	  		  in_diff->ColRange(input_context_indexes_[i][j]*dim_in_, dim_in_).AddMat(1.0, diff_output_.RowRange(i*num_frames, num_frames).ColRange(j*dim_in_, dim_in_));
  }

  void ResetGradient()
  {
      linearity_corr_.SetZero();
      bias_corr_.SetZero();
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff)
  {
		// we use following hyperparameters from the option class
		const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
		const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
		const BaseFloat mmt = opts_.momentum;
		const BaseFloat l2 = opts_.l2_penalty;
		const BaseFloat l1 = opts_.l1_penalty;
		// we will also need the number of frames in the mini-batch
		const int32 num_frames = input.NumRows();
		local_lrate = -lr;
		local_lrate_bias = -lr_bias;

		// compute gradient (incl. momentum)
		linearity_corr_.AddMatMat(1.0, diff_patches_, kTrans, input_patches_, kNoTrans, mmt);
		bias_corr_.AddRowSumMat(1.0, diff_patches_, mmt);

		// l2 regularization
		if (l2 != 0.0) {
		  linearity_.AddMat(-lr*l2*num_frames, linearity_);
		}
		// l1 regularization
		if (l1 != 0.0) {
		  cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
		}
  }

  void UpdateGradient()
  {
	    	// update
		linearity_.AddMat(local_lrate, linearity_corr_);
		bias_.AddVec(local_lrate_bias, bias_corr_);

		// max-norm
		if (max_norm_ > 0.0) {
			CuMatrix<BaseFloat> lin_sqr(linearity_);
			lin_sqr.MulElements(linearity_);
			CuVector<BaseFloat> l2(OutputDim());
			l2.AddColSumMat(1.0, lin_sqr, 0.0);
			l2.ApplyPow(0.5); // we have per-neuron L2 norms
			CuVector<BaseFloat> scl(l2);
			scl.Scale(1.0/max_norm_);
			scl.ApplyFloor(1.0);
			scl.InvertElements();
			linearity_.MulRowsVec(scl); // shink to sphere!
		}
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff_patches_, kTrans, input_patches_, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff_patches_, mmt);
    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr*l2*num_frames, linearity_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr*l1*num_frames, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr_bias, bias_corr_);
    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(linearity_);
      lin_sqr.MulElements(linearity_);
      CuVector<BaseFloat> l2(OutputDim());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5); // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      linearity_.MulRowsVec(scl); // shink to sphere!
    }
  }

  /// Accessors to the component parameters
  const CuVectorBase<BaseFloat>& GetBias() const {
    return bias_;
  }

  void SetBias(const CuVectorBase<BaseFloat>& bias) {
    KALDI_ASSERT(bias.Dim() == bias_.Dim());
    bias_.CopyFromVec(bias);
  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const {
    return linearity_;
  }

  void SetLinearity(const CuMatrixBase<BaseFloat>& linearity) {
    KALDI_ASSERT(linearity.NumRows() == linearity_.NumRows());
    KALDI_ASSERT(linearity.NumCols() == linearity_.NumCols());
    linearity_.CopyFromMat(linearity);
  }

  const CuVectorBase<BaseFloat>& GetBiasCorr() const {
    return bias_corr_;
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() const {
    return linearity_corr_;
  }

  int WeightCopy(void *host, int direction, int copykind)
  {
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

		dim = linearity_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)linearity_.Data());
		src = (void*) (direction==0 ? (char *)linearity_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += linearity_.SizeInBytes();

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

protected:
  int32 num_input_context_;
  int32 num_output_context_;
  int32 num_indexes_;
  int32 dim_in_;
  int32 dim_out_;
  std::vector<std::vector<int32> > input_context_indexes_;

  CuMatrix<BaseFloat> input_patches_;
  CuMatrix<BaseFloat> forward_output_;

  CuMatrix<BaseFloat> diff_patches_;
  CuMatrix<BaseFloat> diff_output_;

  // weights
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;
  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;

  BaseFloat local_lrate;
  BaseFloat local_lrate_bias;
};

} // namespace nnet0
} // namespace kaldi

#endif
