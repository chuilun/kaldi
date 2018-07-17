// nnet0/nnet-affine-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)
// Copyright 2016-2017  AISpeech (author: Tao Xu)

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


#ifndef KALDI_NNET_NNET_AFFINE_TRANSFORM_H_
#define KALDI_NNET_NNET_AFFINE_TRANSFORM_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class AffineTransform : public UpdatableComponent {

	friend class NnetModelSync;
	friend class lm::LmModelSync;

 public:
  AffineTransform(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out), 
      linearity_(dim_out, dim_in), bias_(dim_out),
      linearity_corr_(dim_out, dim_in), bias_corr_(dim_out),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0), fix_(0), bit_(7) 
  { }
  ~AffineTransform()
  { }

  Component* Copy() const { return new AffineTransform(*this); }
  ComponentType GetType() const { return kAffineTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    int32 fix_ = 0;
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
      else if (token == "<FixPoint>") ReadBasicType(is, false, &fix_);
      else if (token == "<Bit>") ReadBasicType(is, false, &bit_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

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
    // fixed point
    if(fix_){
        linearity_fix_= linearity_;
    }

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
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<FixPoint>");
      ReadBasicType(is, binary, &fix_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<Bit>");
      ReadBasicType(is, binary, &bit_);
    }
    // weights
    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    if(fix_){
        linearity_fix_ = linearity_;
        linearity_row_max_.Resize(linearity_.NumRows());
        linearity_fix_.FindRowAbsMax(linearity_row_max_);
        linearity_fix_.DivRowsVec(linearity_row_max_);
        linearity_fix_.ApplyFixed(pow(2, -bit_), fix_);
    }

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
    // weights
    if(fix_) {
        WriteToken(os, binary, "<FixPoint>");
        WriteBasicType(os, binary, fix_);
        WriteToken(os, binary, "<Bit>");
        WriteBasicType(os, binary, bit_);
        linearity_fix_.Write(os, binary);
    } else 
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
    std::string fixparm = fix_ ? std::string("\n  linearity_fix_") + MomentStatistics(linearity_fix_) : "";
    return std::string("\n  linearity") + MomentStatistics(linearity_) + fixparm +
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
    out->AddVecToRows(1.0, bias_, 0.0);

    if(fix_){
        if (input_fix_.NumRows() != in.NumRows())
            input_fix_.Resize(in.NumRows(), in.NumCols());
        input_fix_.CopyFromMat(in);
        row_max_.Resize(in.NumRows(), kSetZero);
        in.FindRowAbsMax(row_max_);
        input_fix_.DivRowsVec(row_max_);
        input_fix_.ApplyFixed(pow(2, -bit_), fix_);
        out->AddMatMat(1.0, input_fix_, kNoTrans, linearity_fix_, kTrans, 0.0);
        out->MulRowsVec(row_max_);
        out->MulColsVec(linearity_row_max_);       
        out->AddVecToRows(1.0, bias_, 1.0); 
    }else{
        // multiply by weights^t
        out->AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
    }

    p_input_ = fix_ ? &input_fix_ : &in;
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
	in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
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
	    linearity_corr_.AddMatMat(1.0, diff, kTrans, *p_input_, kNoTrans, mmt);
	    bias_corr_.AddRowSumMat(1.0, diff, mmt);
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
          if(fix_){
              bias_.ApplyFloor(-8.0);
              bias_.ApplyCeiling(8.0);
              //linearity_.ApplyFloor(-8.0);
              //linearity_.ApplyCeiling(8.0);
              linearity_fix_.CopyFromMat(linearity_);
              linearity_fix_.FindRowAbsMax(linearity_row_max_);
              linearity_fix_.DivRowsVec(linearity_row_max_); 
              linearity_fix_.ApplyFixed(pow(2, -bit_), fix_);
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
    linearity_corr_.AddMatMat(1.0, diff, kTrans, *p_input_, kNoTrans, mmt);
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
    if(fix_){
      bias_.ApplyFloor(-8.0);
      bias_.ApplyCeiling(8.0);
      //linearity_.ApplyFloor(-8.0);
      //linearity_.ApplyCeiling(8.0);
      linearity_fix_.CopyFromMat(linearity_);
      linearity_fix_.FindRowAbsMax(linearity_row_max_);
      linearity_fix_.DivRowsVec(linearity_row_max_); 
      linearity_fix_.ApplyFixed(pow(2, -bit_), fix_);
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

  /// This function is for getting a low-rank approximations of this
   /// AffineComponent by two AffineComponents.
  virtual void LimitRank(BaseFloat threshold, int min_rank, int max_rank,
		  AffineTransform **a, AffineTransform **b) const {
    int32 d = 0;
    BaseFloat sum = 0;
    bool transed = false;

    // We'll limit the rank of just the linear part, keeping the bias vector full.
    Matrix<BaseFloat> M (linearity_);
    if (M.NumRows() < M.NumCols())
    {
    	M.Transpose();
    	transed = true;
    }
    int32 rows = M.NumRows(), cols = M.NumCols(), rc_min = std::min(rows, cols);
    Vector<BaseFloat> s(rc_min);
    Matrix<BaseFloat> U(rows, rc_min), Vt(rc_min, cols);
    // Do the destructive svd M = U diag(s) V^T.  It actually outputs the transpose of V.
    M.DestructiveSvd(&s, &U, &Vt);
    SortSvd(&s, &U, &Vt); // Sort the singular values from largest to smallest.
    BaseFloat old_svd_sum = s.Sum();

    for (d = 0; d < s.Dim(); d++)
    {
    	sum += s(d);
    	if (sum >= old_svd_sum*threshold) break;
    }

    KALDI_ASSERT(d <= InputDim());
    d = d < min_rank ? min_rank : d;
    d = max_rank < d ? max_rank : d;

    U.Resize(rows, d, kCopyData);
    s.Resize(d, kCopyData);
    Vt.Resize(d, cols, kCopyData);
    BaseFloat new_svd_sum = s.Sum();
    KALDI_LOG << "Reduced rank from "
              << rows << "x" << cols <<  " to "
              << rows << "x" << d << " and " << d << "x" << cols
	          << ", SVD sum reduced from " << old_svd_sum << " to " << new_svd_sum;

    if (transed)
    {
	U.Transpose();
	Vt.Transpose();
	U.Swap(&Vt);
    }
    s.ApplySqrt();
    U.MulColsVec(s); // U <-- U diag(s)
    Vt.MulRowsVec(s); // Vt <-- diag(s) Vt.

    //*a = dynamic_cast<AffineTransform*>(this->Copy());
    //*b = dynamic_cast<AffineTransform*>(this->Copy());
    *a = new AffineTransform(Vt.NumCols(), Vt.NumRows());
    *b = new AffineTransform(U.NumCols(), U.NumRows());

    (*a)->bias_.Resize(d, kSetZero);
    (*a)->linearity_ = Vt;

    (*b)->bias_ = this->bias_;
    (*b)->linearity_ = U;
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
  friend class AffinePreconditionedOnlineTransform;
  CuMatrix<BaseFloat> linearity_;
  CuVector<BaseFloat> bias_;
  CuMatrix<BaseFloat> linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  // for fixed point training
  CuVector<BaseFloat> row_max_;
  CuMatrix<BaseFloat> linearity_fix_;
  CuVector<BaseFloat> linearity_row_max_;  
  CuMatrix<BaseFloat> input_fix_;
  const CuMatrixBase<BaseFloat> *p_input_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;

  BaseFloat local_lrate;
  BaseFloat local_lrate_bias;
  int32 fix_ ;
  int32 bit_;
};

} // namespace nnet0
} // namespace kaldi

#endif
