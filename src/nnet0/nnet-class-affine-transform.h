	// nnet0/nnet-class-affine-transform.h

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


#ifndef KALDI_NNET_NNET_CLASS_AFFINE_TRANSFORM_H_
#define KALDI_NNET_NNET_CLASS_AFFINE_TRANSFORM_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class ClassAffineTransform : public UpdatableComponent {

	friend class NnetModelSync;
    friend class lm::LmModelSync;

 public:
	ClassAffineTransform(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0),
	  max_norm_(0.0), num_class_(0)
  { }
  ~ClassAffineTransform()
  { }

  Component* Copy() const { return new ClassAffineTransform(*this); }
  ComponentType GetType() const { return kClassAffineTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    int32 num_class = 0;
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
      else if (token == "<Class>") ReadBasicType(is, false, &num_class);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    KALDI_ASSERT(num_class > 0);

    //
    // initialize
    //
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        if (param_range == 0.0)
        	mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
	else
		mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
      }
    }
    linearity_ = mat;
    //
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;

    //
    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    num_class_ = num_class;
    //

    class_boundary_.resize(num_class_+1);
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
      ExpectToken(is, binary, "<ClassSize>");
      ReadBasicType(is, binary, &num_class_);
    }
    // weights
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    WriteToken(os, binary, "<ClassSize>");
    WriteBasicType(os, binary, num_class_);
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
    // precopy bias
    // out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    // out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);

    for (int p = 0; p < input_patches_.size(); p++)
    {
        delete input_patches_[p];   
        delete output_patches_[p];  
    }

    CuArray<int32> idx(sortedclass_id_index_);
    input_sorted_.Resize(in.NumRows(), in.NumCols(), kUndefined);

	input_sorted_.CopyRows(in, idx);

    int size = idx.Dim();
    int beg = 0, cid, clen;
    input_patches_.clear();
    output_patches_.clear();
    updateclass_linearity_.clear();
    updateclass_bias_.clear();

    for (int i = 1; i <= size; i++)
    {
    	if (i == size || sortedclass_id_[i] != sortedclass_id_[i-1])
    	{
    		cid = sortedclass_id_[i-1];
    		clen = class_boundary_[cid+1] - class_boundary_[cid];
    		input_patches_.push_back(new CuSubMatrix<BaseFloat>(input_sorted_.RowRange(beg, i-beg)));
    		updateclass_linearity_.push_back(class_linearity_[cid]);
    		updateclass_bias_.push_back(class_bias_[cid]);
    		output_patches_.push_back(new CuSubMatrix<BaseFloat>(out->Range(beg, i-beg, class_boundary_[cid], clen)));
    		beg = i;
    	}
    }
    // class
    clen = output_dim_ - class_boundary_.back();
    input_patches_.push_back(new CuSubMatrix<BaseFloat>(input_sorted_.RowRange(0, input_sorted_.NumRows())));
	updateclass_linearity_.push_back(class_linearity_.back());
	updateclass_bias_.push_back(class_bias_.back());
	output_patches_.push_back(new CuSubMatrix<BaseFloat>(out->ColRange(class_boundary_.back(), clen)));

#if HAVE_CUDA == 1
    SetStream(input_patches_, streamlist_);
   	SetStream(output_patches_, streamlist_);
    SetStream(updateclass_linearity_, streamlist_);
   	SetStream(updateclass_bias_, streamlist_);
#endif

    AddVecToRowsStreamed(static_cast<BaseFloat>(1.0f), output_patches_, updateclass_bias_, static_cast<BaseFloat>(0.0f));
    AddMatMatStreamed(static_cast<BaseFloat>(1.0f), output_patches_, input_patches_, kNoTrans,
    									updateclass_linearity_, kTrans, static_cast<BaseFloat>(1.0f));

#if HAVE_CUDA == 1
    ResetStream(input_patches_);
    ResetStream(output_patches_);
    ResetStream(updateclass_linearity_);
    ResetStream(updateclass_bias_);
#endif
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
	// in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);

    for (int p = 0; p < in_diff_patches_.size(); p++)
    {
        delete in_diff_patches_[p];   
        delete out_diff_patches_[p];  
    }

	  input_diff_sorted_.Resize(in_diff->NumRows(), in_diff->NumCols(), kUndefined);
	  in_diff_patches_.clear();
	  out_diff_patches_.clear();
	  updateclass_linearity_corr_.clear();
	  updateclass_bias_corr_.clear();


	    int size = sortedclass_id_.size();
	    int beg = 0, cid, clen;

	    for (int i = 1; i <= size; i++)
	    {
	    	if (i == size || sortedclass_id_[i] != sortedclass_id_[i-1])
	    	{
	    		cid = sortedclass_id_[i-1];
	    		clen = class_boundary_[cid+1] - class_boundary_[cid];
	    		in_diff_patches_.push_back(new CuSubMatrix<BaseFloat>(input_diff_sorted_.RowRange(beg, i-beg)));
	    		updateclass_linearity_corr_.push_back(class_linearity_corr_[cid]);
	    		updateclass_bias_corr_.push_back(class_bias_corr_[cid]);
	    		out_diff_patches_.push_back(new CuSubMatrix<BaseFloat>(out_diff.Range(beg, i-beg, class_boundary_[cid], clen)));
	    		beg = i;
	    	}
	    }

        updateclass_linearity_.resize(out_diff_patches_.size());
#if HAVE_CUDA == 1
	    SetStream(in_diff_patches_, streamlist_);
	    SetStream(out_diff_patches_, streamlist_);
	    SetStream(updateclass_linearity_, streamlist_);
#endif

	    AddMatMatStreamed(static_cast<BaseFloat>(1.0f), in_diff_patches_, out_diff_patches_, kNoTrans,
	    												updateclass_linearity_, kNoTrans, static_cast<BaseFloat>(0.0f));

#if HAVE_CUDA == 1
	    ResetStream(in_diff_patches_);
	    ResetStream(out_diff_patches_);
	    ResetStream(updateclass_linearity_);
#endif

	    // class
	    clen = output_dim_ - class_boundary_.back();
	    CuSubMatrix<BaseFloat> *out_diff_class = new CuSubMatrix<BaseFloat>(out_diff.ColRange(class_boundary_.back(), clen));
	    input_diff_sorted_.AddMatMat(1.0, *out_diff_class, kNoTrans, *class_linearity_[num_class_], kNoTrans, 1.0);
        //out_diff_class->SetZero();
        //delete out_diff_class;

        // restore input
        CuArray<int32> idx(sortedclass_id_reindex_);
	    in_diff->CopyRows(input_diff_sorted_, idx);

        // for last class gradient
	    updateclass_linearity_corr_.push_back(class_linearity_corr_.back());
	    updateclass_bias_corr_.push_back(class_bias_corr_.back());
	    updateclass_linearity_.push_back(class_linearity_.back());
	    out_diff_patches_.push_back(out_diff_class);
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff)
  {
	    // we use following hyperparameters from the option class
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
	    const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
	    const BaseFloat mmt = opts_.momentum;
	    const BaseFloat l2 = opts_.l2_penalty;
	    //const BaseFloat l1 = opts_.l1_penalty;
	    // we will also need the number of frames in the mini-batch
	    const int32 num_frames = input.NumRows();
		local_lrate = -lr;
		local_lrate_bias = -lr_bias;


#if HAVE_CUDA == 1
	    SetStream(updateclass_linearity_corr_, streamlist_);
	   	SetStream(out_diff_patches_, streamlist_);
	    SetStream(input_patches_, streamlist_);
	    SetStream(updateclass_bias_corr_, streamlist_);
	    SetStream(updateclass_linearity_, streamlist_);
#endif

		AddMatMatStreamed(static_cast<BaseFloat>(1.0f), updateclass_linearity_corr_, out_diff_patches_, kTrans,
																input_patches_, kNoTrans, static_cast<BaseFloat>(mmt));
		AddRowSumMatStreamed(static_cast<BaseFloat>(1.0f), updateclass_bias_corr_, out_diff_patches_, mmt);

        // l2 regularization
        if (l2 != 0.0) {
            AddMatStreamed(-lr*l2*num_frames, updateclass_linearity_, updateclass_linearity_);
        }

#if HAVE_CUDA == 1
		ResetStream(updateclass_linearity_corr_);
		ResetStream(out_diff_patches_);
		ResetStream(input_patches_);
		ResetStream(updateclass_bias_corr_);
	    ResetStream(updateclass_linearity_);
#endif
  }

  void UpdateGradient()
  {
	    	// update
	      //linearity_.AddMat(local_lrate, linearity_corr_);
	      //bias_.AddVec(local_lrate_bias, bias_corr_);

#if HAVE_CUDA == 1
	  	  SetStream(updateclass_linearity_, streamlist_);
	  	  SetStream(updateclass_bias_, streamlist_);
#endif
	  	  AddMatStreamed(local_lrate, updateclass_linearity_, updateclass_linearity_corr_);
	  	  AddVecStreamed(local_lrate_bias, updateclass_bias_, updateclass_bias_corr_);

#if HAVE_CUDA == 1
	  	  ResetStream(updateclass_bias_);
	  	  ResetStream(updateclass_linearity_);
#endif
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
    linearity_corr_.AddMatMat(1.0, diff, kTrans, input, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff, mmt);

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

  void SetClassBoundary(const std::vector<int32>& class_boundary)
  {
      KALDI_ASSERT(class_boundary.size() == num_class_ + 1);
	  class_boundary_ = class_boundary;

	  int size = class_boundary_.size(), len;
	  class_linearity_.resize(size);
      class_linearity_corr_.resize(size);
      class_bias_.resize(size);
      class_bias_corr_.resize(size);

	  for (int i = 0; i < size; i++)
	  {
		  len = (i==size-1) ? output_dim_ - class_boundary_[size-1] : class_boundary_[i+1] - class_boundary_[i];
		  class_linearity_[i] = new CuSubMatrix<BaseFloat>(linearity_.RowRange(class_boundary_[i], len));
		  class_linearity_corr_[i] = new CuSubMatrix<BaseFloat>(linearity_corr_.RowRange(class_boundary_[i], len));
		  class_bias_[i] = new CuSubVector<BaseFloat>(bias_.Range(class_boundary_[i], len));
		  class_bias_corr_[i] = new CuSubVector<BaseFloat>(bias_corr_.Range(class_boundary_[i], len));
	  }

#if HAVE_CUDA == 1
	  int32 num_class = class_boundary.size()-1;
	  streamlist_.resize(num_class+1);
	  for (int i = 0; i < num_class+1; i++)
		  cudaStreamCreateWithFlags(&streamlist_[i], cudaStreamNonBlocking);
#endif
  }

  std::vector<int32> GetClassBoundary()
  {
	  return this->class_boundary_;
  }

  void SetUpdateClassId(const std::vector<int32>& sortedclass_id, const std::vector<int32>& sortedclass_id_index, std::vector<int32> &sortedclass_id_reindex)
  {
	  sortedclass_id_ = sortedclass_id;
	  sortedclass_id_index_ = sortedclass_id_index;
	  sortedclass_id_reindex_ = sortedclass_id_reindex;
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

  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;

  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  CuMatrix<BaseFloat> input_sorted_;
  CuMatrix<BaseFloat> input_diff_sorted_;

  std::vector<CuSubMatrix<BaseFloat>* > class_linearity_;
  std::vector<CuSubVector<BaseFloat>* > class_bias_;

  std::vector<CuSubMatrix<BaseFloat>* > class_linearity_corr_;
  std::vector<CuSubVector<BaseFloat>* > class_bias_corr_;

  std::vector<CuSubMatrix<BaseFloat>* > updateclass_linearity_;
  std::vector<CuSubVector<BaseFloat>* > updateclass_bias_;

  std::vector<CuSubMatrix<BaseFloat>* > updateclass_linearity_corr_;
  std::vector<CuSubVector<BaseFloat>* > updateclass_bias_corr_;

  std::vector<CuSubMatrix<BaseFloat>* > input_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > output_patches_;

  std::vector<CuSubMatrix<BaseFloat>* > in_diff_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > out_diff_patches_;

  std::vector<int32> class_boundary_;
  //std::vector<int32> updateclass_id_;
  std::vector<int32> sortedclass_id_;
  std::vector<int32> sortedclass_id_index_;
  std::vector<int32> sortedclass_id_reindex_;


  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;

  BaseFloat local_lrate;
  BaseFloat local_lrate_bias;

  BaseFloat max_norm_;
  int32 num_class_;

#if HAVE_CUDA == 1
  std::vector<cudaStream_t > streamlist_;
#endif

};

} // namespace nnet0
} // namespace kaldi

#endif
