// nnet0/nnet-max-pooling-2d-component-fast.h

// Copyright 2014  Brno University of Technology (author: Karel Vesely),
//                 Johns Hopkins University (author: Sri Harish Mallidi)

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


#ifndef KALDI_NNET_NNET_MAX_POOLING_2D_COMPONENT_FAST_H_
#define KALDI_NNET_NNET_MAX_POOLING_2D_COMPONENT_FAST_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet0 {

/**
 * MaxPoolingComponent :
 * The input/output matrices are split to submatrices with width 'pool_stride_'.
 * The pooling is done over 3rd axis, of the set of 2d matrices.
 * Our pooling supports overlaps, overlaps occur when (pool_step_ < pool_size_).
 */
class MaxPooling2DComponentFast : public Component {
 public:
  MaxPooling2DComponentFast(int32 dim_in, int32 dim_out)
      : Component(dim_in, dim_out),
        fmap_x_len_(0), fmap_y_len_(0),
        pool_x_len_(0), pool_y_len_(0), pool_x_step_(0), pool_y_step_(0)
  { }
  ~MaxPooling2DComponentFast()
  { }

  Component* Copy() const { return new MaxPooling2DComponentFast(*this); }
  ComponentType GetType() const { return kMaxPooling2DComponentFast; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<FmapXLen>") ReadBasicType(is, false, &fmap_x_len_);
      else if (token == "<FmapYLen>") ReadBasicType(is, false, &fmap_y_len_);
      else if (token == "<PoolXLen>") ReadBasicType(is, false, &pool_x_len_);
      else if (token == "<PoolYLen>") ReadBasicType(is, false, &pool_y_len_);
      else if (token == "<PoolXStep>") ReadBasicType(is, false, &pool_x_step_);
      else if (token == "<PoolYStep>") ReadBasicType(is, false, &pool_y_step_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (FmapXLen|FmapYLen|PoolXLen|PoolYLen|PoolXStep|PoolYStep)";
      is >> std::ws;  // eat-up whitespace
    }
    // check
    KALDI_ASSERT(fmap_x_len_ * fmap_y_len_ * pool_x_len_ * pool_y_len_ * pool_x_step_ * pool_y_step_  != 0);
  }

  void ReadData(std::istream &is, bool binary) {
    // pooling hyperparameters
    ExpectToken(is, binary, "<FmapXLen>");
    ReadBasicType(is, binary, &fmap_x_len_);
    ExpectToken(is, binary, "<FmapYLen>");
    ReadBasicType(is, binary, &fmap_y_len_);
    ExpectToken(is, binary, "<PoolXLen>");
    ReadBasicType(is, binary, &pool_x_len_);
    ExpectToken(is, binary, "<PoolYLen>");
    ReadBasicType(is, binary, &pool_y_len_);
    ExpectToken(is, binary, "<PoolXStep>");
    ReadBasicType(is, binary, &pool_x_step_);
    ExpectToken(is, binary, "<PoolYStep>");
    ReadBasicType(is, binary, &pool_y_step_);

    //
    // Sanity checks:
    //
    // input sanity checks
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    KALDI_LOG << "num_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    // KALDI_ASSERT((fmap_x_len_ - pool_x_len_) % (pool_x_step_) == 0);
    // KALDI_ASSERT((fmap_y_len_ - pool_y_len_) % (pool_y_step_) == 0);
    int32 out_fmap_x_len = (fmap_x_len_ - pool_x_len_)/pool_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - pool_y_len_)/pool_y_step_ + 1;
    //    int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // output sanity checks
    KALDI_ASSERT(output_dim_ % (out_fmap_x_len * out_fmap_y_len)  == 0);
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    KALDI_ASSERT(num_input_fmaps == num_output_fmaps);


    //
        // here we note how many diff matrices are summed for each input patch,
        // this metainfo will be used to divide diff of patches
        // used in more than one pool.
        //

    int32 input_fmap_size = fmap_x_len_ * fmap_y_len_;

    Vector<BaseFloat> patch_summands(input_fmap_size, kSetZero);
    BaseFloat *data_summands = patch_summands.Data();
    int out_fmap_cnt = 0;
    for (int32 m = 0; m < fmap_x_len_-pool_x_len_+1; m = m+pool_x_step_) {
      for (int32 n = 0; n < fmap_y_len_-pool_y_len_+1; n = n+pool_y_step_) {
        int32 st = 0;
        st = (m*fmap_y_len_+n)*num_input_fmaps;

        for (int32 i = 0; i < pool_x_len_; i++) {
          for (int32 j = 0; j < pool_y_len_; j++) {
            int32 c = 0;
            c = st + i * (num_input_fmaps * fmap_y_len_)
                   + j * num_input_fmaps;

            data_summands[c/num_input_fmaps] += 1;
          }
        }
        out_fmap_cnt++;
      }
    }

    // patch at least in one pool
    // aviod divide 0 produce inf
    patch_summands.ApplyFloor(1.0);
    patch_summands.InvertElements();
    Vector<BaseFloat> tmp(num_input_fmaps*input_fmap_size, kSetZero);
    BaseFloat *data_tmp = tmp.Data();
    for (int32 i=0; i < input_fmap_size; i++)
    	for (int32 j=0; j < num_input_fmaps; j++)
    		data_tmp[i*num_input_fmaps+j] = data_summands[i];
    		//data_tmp[i*num_input_fmaps+j] = data_summands[i] > 1.0 ? 0 : data_summands[i]; // aviod divide 0 produce inf
    patch_summands_.Resize(num_input_fmaps*input_fmap_size, kSetZero);
    patch_summands_.CopyFromVec(tmp);
  }

  void WriteData(std::ostream &os, bool binary) const {
    // pooling hyperparameters
    WriteToken(os, binary, "<FmapXLen>");
    WriteBasicType(os, binary, fmap_x_len_);
    WriteToken(os, binary, "<FmapYLen>");
    WriteBasicType(os, binary, fmap_y_len_);
    WriteToken(os, binary, "<PoolXLen>");
    WriteBasicType(os, binary, pool_x_len_);
    WriteToken(os, binary, "<PoolYLen>");
    WriteBasicType(os, binary, pool_y_len_);
    WriteToken(os, binary, "<PoolXStep>");
    WriteBasicType(os, binary, pool_x_step_);
    WriteToken(os, binary, "<PoolYStep>");
    WriteBasicType(os, binary, pool_y_step_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);

    out->Set(-1e20);
    out->MaxPoolingForward(in, num_input_fmaps, fmap_x_len_, fmap_y_len_, pool_x_len_, pool_y_len_, pool_x_step_, pool_y_step_);
    //  CuVector<BaseFloat> tmp(out->Row(0));
    //	tmp.Write(std::cout, false);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    //int32 inp_fmap_size = fmap_x_len_ * fmap_y_len_;

    in_diff->SetZero();  // reset

    in_diff->MaxPoolingBackward(in, out, out_diff,
    		num_input_fmaps, fmap_x_len_, fmap_y_len_, pool_x_len_, pool_y_len_, pool_x_step_, pool_y_step_);

    // divide diff by #summands (compensate for patches used in more pools)
    in_diff->MulColsVec(patch_summands_);

  }

 private:
  int32 fmap_x_len_, fmap_y_len_,
    pool_x_len_, pool_y_len_,
    pool_x_step_, pool_y_step_;
  CuVector<BaseFloat> patch_summands_;
};

}  // namespace nnet0
}  // namespace kaldi

#endif
