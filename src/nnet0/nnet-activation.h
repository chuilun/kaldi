// nnet0/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#if HAVE_CUDA == 1
#include <cudnn.h>
#endif

#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet0 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
};

class CBSoftmax : public Component {
 public:
	CBSoftmax(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
	~CBSoftmax()
  { }

  Component* Copy() const { return new CBSoftmax(*this); }
  ComponentType GetType() const { return kCBSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    int size = updateclass_id_.size();
    int beg, cid, clen;

    for (int p = 0; p < input_patches_.size(); p++)
    {
        delete input_patches_[p];
        delete output_patches_[p];
        delete frame_zt_patches_[p];
    }

    input_patches_.clear();
    output_patches_.clear();
    frame_zt_patches_.clear();
    frame_zt_.Resize(size*2, kUndefined);

    beg = 0;
    for (int i = 1; i <= size; i++)
    {
    	if (i == size || updateclass_id_[i] != updateclass_id_[i-1])
    	{
    		cid = updateclass_id_[i-1];
    		clen = class_boundary_[cid+1] - class_boundary_[cid];
    		input_patches_.push_back(new CuSubMatrix<BaseFloat>(in.Range(beg, i-beg, class_boundary_[cid], clen)));
    		output_patches_.push_back(new CuSubMatrix<BaseFloat>(out->Range(beg, i-beg, class_boundary_[cid], clen)));
    		frame_zt_patches_.push_back(new CuSubVector<BaseFloat>(frame_zt_.Range(beg, i-beg)));
    		beg = i;
    	}
    }

    // class
    clen = output_dim_ - class_boundary_.back();
    input_patches_.push_back(new CuSubMatrix<BaseFloat>(in.ColRange(class_boundary_.back(), clen)));
    output_patches_.push_back(new CuSubMatrix<BaseFloat>(out->ColRange(class_boundary_.back(), clen)));
    frame_zt_patches_.push_back(new CuSubVector<BaseFloat>(frame_zt_.Range(size, size)));

#if HAVE_CUDA == 1
    SetStream(input_patches_, streamlist_);
   	SetStream(output_patches_, streamlist_);
#endif

	ApplySoftMaxPerRowStreamed(output_patches_, input_patches_, &frame_zt_patches_);

#if HAVE_CUDA == 1
	ResetStream(input_patches_);
	ResetStream(output_patches_);
#endif

  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function,
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with
    // respect to activations of last layer neurons)
    // in_diff->CopyFromMat(out_diff);

    int size = updateclass_id_.size();
    int beg, cid, clen;

    indiff_patches_.clear();
    outdiff_patches_.clear();
    beg = 0;
    for (int i = 1; i <= size; i++)
    {
    	if (i == size || updateclass_id_[i] != updateclass_id_[i-1])
    	{
    		cid = updateclass_id_[i-1];
    		clen = class_boundary_[cid+1] - class_boundary_[cid];
    		indiff_patches_.push_back(new CuSubMatrix<BaseFloat>(in_diff->Range(beg, i-beg, class_boundary_[cid], clen)));
    		outdiff_patches_.push_back(new CuSubMatrix<BaseFloat>(out_diff.Range(beg, i-beg, class_boundary_[cid], clen)));
    		beg = i;
    	}
    }

    // class
    clen = output_dim_ - class_boundary_.back();
    indiff_patches_.push_back(new CuSubMatrix<BaseFloat>(in_diff->ColRange(class_boundary_.back(), clen)));
    outdiff_patches_.push_back(new CuSubMatrix<BaseFloat>(out_diff.ColRange(class_boundary_.back(), clen)));

#if HAVE_CUDA == 1
    SetStream(indiff_patches_, streamlist_);
   	SetStream(outdiff_patches_, streamlist_);
#endif

    CopyFromMatStreamed(outdiff_patches_, indiff_patches_);

#if HAVE_CUDA == 1
	ResetStream(indiff_patches_);
	ResetStream(outdiff_patches_);
#endif

    for (int p = 0; p < indiff_patches_.size(); p++)
    {
        delete indiff_patches_[p];
        delete outdiff_patches_[p];
    }

  }

  void SetClassBoundary(const std::vector<int32>& class_boundary)
  {
	  class_boundary_ = class_boundary;

#if HAVE_CUDA == 1
	  int32 num_class = class_boundary.size()-1;
	  streamlist_.resize(num_class+1);
	  for (int i = 0; i < num_class+1; i++)
		  cudaStreamCreateWithFlags(&streamlist_[i], cudaStreamNonBlocking);
#endif
  }

  void SetUpdateClassId(const std::vector<int32>& updateclass_id)
  {
	  updateclass_id_ = updateclass_id;
  }

  CuVector<BaseFloat>* GetZt()
  {
	  return &frame_zt_;
  }

  std::vector<CuSubVector<BaseFloat>* >* GetZtPatches()
  {
	  return &frame_zt_patches_;
  }

 private:
  std::vector<int32> class_boundary_;
  std::vector<int32> updateclass_id_;
  std::vector<CuSubMatrix<BaseFloat>* > input_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > output_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > indiff_patches_;
  std::vector<CuSubMatrix<BaseFloat>* > outdiff_patches_;

  // constant normalizing
  CuVector<BaseFloat> frame_zt_;
  std::vector<CuSubVector<BaseFloat>* > frame_zt_patches_;

#if HAVE_CUDA == 1
  std::vector<cudaStream_t > streamlist_;
#endif
};


class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ",:", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim()); 
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j)
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_dims[bl]);
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::string Info() const {
    return "\n  softmax-dims " + ToString(block_dims);
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};




class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};

class Relu : public Component {
 public:
        Relu(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { } 
  ~Relu()
  { } 

  Component* Copy() const { return new Relu(*this); }
  ComponentType GetType() const { return kRelu; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (x >= 0 ? x : 0.0)
          out->CopyFromMat(in);
          out->ApplyFloor(0.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
          // Now in_deriv(i, j) equals (out_value(i, j) > 0.0 ? 1.0 : 0.0),
          // which is the derivative of the nonlinearity (well, except at zero
          // where it's undefined).
          in_diff->CopyFromMat(out);
          in_diff->ApplyHeaviside();
          in_diff->MulElements(out_diff);
  }
};

#if HAVE_CUDA == 1
class CudnnRelu : public Component {

    public:
        CudnnRelu(int32 dim_in, int32 dim_out):Component(dim_in, dim_out),initialized_(false),mode_(CUDNN_ACTIVATION_RELU),ceil_(50.0)
        {}
        ~CudnnRelu()
        {
            if(initialized_){

                CU_SAFE_CALL(cudnnDestroyTensorDescriptor(shape_desc_));
                CU_SAFE_CALL(cudnnDestroyActivationDescriptor(desc_));

                cudaStreamDestroy(stream_);
                cudnnDestroy(handle_);
            }
        }
        Component* Copy() const {return new CudnnRelu(*this);}
        ComponentType GetType() const {return kCudnnRelu;}

        void Init(int32 dim, int32 batch_size){
             
            cudaStreamCreate(&stream_);
            cudnnCreate(&handle_);
            cudnnSetStream(handle_, stream_); 
        
            CU_SAFE_CALL(cudnnCreateActivationDescriptor(&desc_));
            CU_SAFE_CALL(cudnnSetActivationDescriptor(desc_, mode_, CUDNN_NOT_PROPAGATE_NAN, ceil_));
            CU_SAFE_CALL(cudnnCreateTensorDescriptor(&shape_desc_));
            CU_SAFE_CALL(cudnnSetTensor4dDescriptor(shape_desc_,
                                                CUDNN_TENSOR_NCHW,
                                                CUDNN_DATA_FLOAT,
                                                batch_size,
                                                dim,
                                                1,
                                                1));
        }
        void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out){

            if(!initialized_){
                Init(in.NumCols(), in.NumRows());
                initialized_ = true ;
            }

            BaseFloat alpha = 1.0f ;
            BaseFloat beta = 0.0f ;
            const BaseFloat* in_ptr = in.Data();
            BaseFloat* out_ptr = out->Data();
        
            CU_SAFE_CALL(cudnnActivationForward(handle_,
                                            desc_,  
                                            &alpha,
                                            shape_desc_,
                                            in_ptr,
                                            &beta,
                                            shape_desc_,
                                            out_ptr));
        }

        

        void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                              const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
            BaseFloat alpha = 1.0f ;
            BaseFloat beta = 0.0f ;
            const BaseFloat* out_ptr = out.Data();
            const BaseFloat* in_ptr = in.Data();
            const BaseFloat* out_diff_ptr = out_diff.Data();
            BaseFloat* in_diff_ptr = in_diff->Data();
            
            CU_SAFE_CALL(cudnnActivationBackward(handle_,
                                             desc_,
                                             &alpha,
                                             shape_desc_,
                                             out_ptr,
                                             shape_desc_,
                                             out_diff_ptr,
                                             shape_desc_,
                                             in_ptr,
                                             &beta,
                                             shape_desc_,
                                             in_diff_ptr));
        }
    private:
        bool initialized_ ;
        cudnnActivationMode_t mode_ ;
        cudnnTensorDescriptor_t shape_desc_ ;
        cudnnActivationDescriptor_t desc_ ;
        BaseFloat ceil_ ;
        cudnnHandle_t handle_ ;
        cudaStream_t stream_ ;

};
#endif


class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
      is >> std::ws; // eat-up whitespace
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



} // namespace nnet0
} // namespace kaldi

#endif

