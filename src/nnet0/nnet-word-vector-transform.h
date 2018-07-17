	// nnet0/nnet-word-vector-transform.h

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


#ifndef KALDI_NNET_WORD_VECTOR_TRANSFORM_H_
#define KALDI_NNET_WORD_VECTOR_TRANSFORM_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class WordVectorTransform : public UpdatableComponent {

	friend class NnetModelSync;
	friend class lm::LmModelSync;

 public:
	WordVectorTransform(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out), 
	  learn_rate_coef_(1.0)
  { }
  ~WordVectorTransform()
  { }

  Component* Copy() const { return new WordVectorTransform(*this); }
  ComponentType GetType() const { return kWordVectorTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0;
    int32 vocab_size = 0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<VocabSize>") ReadBasicType(is, false, &vocab_size);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    KALDI_ASSERT(vocab_size > 0);

    //
    learn_rate_coef_ = learn_rate_coef;
    vocab_size_ = vocab_size; // wordvector length: output_dim_
    //

    //
    // initialize
    //
    Matrix<BaseFloat> mat(vocab_size_, output_dim_);
    for (int32 r=0; r<vocab_size_; r++) {
      for (int32 c=0; c<output_dim_; c++) {
        if (param_range == 0.0)
        	mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
        else
        	mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
      }
    }
    wordvector_ = mat;

  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<VocabSize>");
      ReadBasicType(is, binary, &vocab_size_);
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
    }
    // weights
    wordvector_.Read(is, binary);

    KALDI_ASSERT(wordvector_.NumRows() == vocab_size_);
    KALDI_ASSERT(wordvector_.NumCols() == output_dim_);

  }

  void WriteData(std::ostream &os, bool binary) const {
	WriteToken(os, binary, "<VocabSize>");
	WriteBasicType(os, binary, vocab_size_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    // weights
    wordvector_.Write(os, binary);
  }

  int32 NumParams() const { return wordvector_.NumRows()*wordvector_.NumCols(); }
  
  int32 GetDim() const { return wordvector_.SizeInBytes()/sizeof(BaseFloat); }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 wordvector_num_elem = wordvector_.NumRows() * wordvector_.NumCols();
    wei_copy->Range(0,wordvector_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(wordvector_));
  }
  
  std::string Info() const {
    return std::string("\n  wordvector") + MomentStatistics(wordvector_);
  }

  std::string InfoGradient() const {
    return std::string("\n  wordvector_grad") + MomentStatistics(wordvector_corr_) +
           ", lr-coef " + ToString(learn_rate_coef_);
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	wordid_.Resize(in.NumRows(), kUndefined);
	CuMatrix<BaseFloat> tmp(1, in.NumRows());
	tmp.CopyFromMat(in, kTrans);
	tmp.CopyRowToVecId(wordid_);
    //std::vector<int32> host(wordid_.Dim());
    //wordid_.CopyToVec(&host);
	out->CopyRows(wordvector_, wordid_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by weights
    //in_diff->AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff)
  {
	    // we use following hyperparameters from the option class
	    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
	    //const BaseFloat mmt = opts_.momentum;
	    //const BaseFloat l2 = opts_.l2_penalty;
	    //const BaseFloat l1 = opts_.l1_penalty;
	    // we will also need the number of frames in the mini-batch
	    //const int32 num_frames = input.NumRows();
		local_lrate = -lr;

	    // compute gradient (incl. momentum)


		for (int p = 0; p < diff_patches_.size(); p++)
    		{
        		delete diff_patches_[p];   
        		delete update_wordvector_patches_[p];  
    		}
		
		// sort word vector error
		wordvector_corr_.Resize(diff.NumRows(), diff.NumCols(), kUndefined);
		CuArray<int32> idx(sortedword_id_index_);
		wordvector_corr_.CopyRows(diff, idx);
		
		diff_patches_.clear();
		update_wordvector_patches_.clear();

	  	int size = sortedword_id_.size();
	  	int beg = 0, wordid = 0;

	  	for (int i = 1; i <= size; i++)
	  	{
	  		if (i == size || sortedword_id_[i] != sortedword_id_[i-1])
	  		{
	  			wordid = sortedword_id_[i-1];
	  			diff_patches_.push_back(new CuSubMatrix<BaseFloat>(wordvector_corr_.RowRange(beg, i-beg)));
	  			update_wordvector_patches_.push_back(new CuSubVector<BaseFloat>(wordvector_.Row(wordid)));
	  			beg = i;
	  		}
	  	}
  }

  void UpdateGradient()
  {
	    // update
#if HAVE_CUDA == 1
	  	SetStream(update_wordvector_patches_, streamlist_);
#endif
	  	AddRowSumMatStreamed(local_lrate, update_wordvector_patches_, diff_patches_, static_cast<BaseFloat>(1.0f));
#if HAVE_CUDA == 1
	  	ResetStream(update_wordvector_patches_);
#endif
	  	//wordvector_.AddMatToRows(local_lrate, wordvector_corr_, wordid_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    //const BaseFloat mmt = opts_.momentum;
    //const BaseFloat l2 = opts_.l2_penalty;
    //const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    //const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)

    wordvector_corr_ = diff;
    wordvector_.AddMatToRows(lr, wordvector_corr_, wordid_);
  }

  void SetUpdateWordId(const std::vector<int32>& sorted_id, const std::vector<int32>& sorted_id_index)
  {
	  sortedword_id_ = sorted_id;
	  sortedword_id_index_ = sorted_id_index;

#if HAVE_CUDA == 1
	  int size = sortedword_id_.size();
	  if (size > streamlist_.size())
	  {
	  	streamlist_.resize(size);
	  	for (int i = 0; i < size; i++)
		  	cudaStreamCreateWithFlags(&streamlist_[i], cudaStreamNonBlocking);
	  }
#endif

  }

  const CuMatrixBase<BaseFloat>& GetLinearity() const {
    return wordvector_;
  }

  void SetLinearity(const CuMatrixBase<BaseFloat>& wordvector) {
    KALDI_ASSERT(wordvector_.NumRows() == wordvector.NumRows());
    KALDI_ASSERT(wordvector_.NumCols() == wordvector.NumCols());
    wordvector_.CopyFromMat(wordvector);
  }

  const CuMatrixBase<BaseFloat>& GetLinearityCorr() const {
    return wordvector_corr_;
  }


  int WeightCopy(void *host, int direction, int copykind)
  {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        Timer tim;

        int32 dst_pitch, src_pitch, width;
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

		dim = wordvector_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)wordvector_.Data());
		src = (void*) (direction==0 ? (char *)wordvector_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += wordvector_.SizeInBytes();


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

  CuMatrix<BaseFloat> wordvector_;

  CuMatrix<BaseFloat> wordvector_corr_;

  CuArray<MatrixIndexT> wordid_;

  BaseFloat learn_rate_coef_;

  BaseFloat local_lrate;

  int32 vocab_size_;

  std::vector<int32> sortedword_id_;
  std::vector<int32> sortedword_id_index_;
  std::vector<CuSubVector<BaseFloat>* > update_wordvector_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > diff_patches_;

#if HAVE_CUDA == 1
  std::vector<cudaStream_t > streamlist_;
#endif
};

} // namespace nnet0
} // namespace kaldi

#endif
