// nnet/nnet-cudnn-convolutional-2d-component.h
//
// Copyright 2015-2016  Shanghai Jiao Tong University (author: Tao Xu)
// Copyright 2015-2016  Shanghai Jiao Tong University (author: Wei Deng)
// //
//
// // See ../../COPYING for clarification regarding multiple authors
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //  http://www.apache.org/licenses/LICENSE-2.0
// //
// // THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// // KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// // WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// // MERCHANTABLITY OR NON-INFRINGEMENT.
// // See the Apache 2 License for the specific language governing permissions and
// // limitations under the License.

#ifndef KALDI_NNET_CUDNN_CONVOLUTIONAL_2D_COMPONENT_H_
#define KALDI_NNET_CUDNN_CONVOLUTIONAL_2D_COMPONENT_H_ 

//#include <cudnn.h>
#if HAVE_CUDA == 1
#include "nnet/nnet-component.h"
#include "base/kaldi-utils.h"
namespace kaldi{
namespace nnet0{

class CudnnConvolutional2DComponent : public UpdatableComponent {
friend class NnetModelSync;
public:
    CudnnConvolutional2DComponent(int32 dim_in, int32 dim_out):UpdatableComponent(dim_in, dim_out),fmap_x_len_(0),fmap_y_len_(0),
    filt_x_len_(0), filt_y_len_(0), filt_x_step_(0), filt_y_step_(0),pad_x_len_(0),pad_y_len_(0),num_output_fmaps_(0),num_input_fmaps_(0),
    out_fmap_x_len_(0), out_fmap_y_len_(0),learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), initialized_(false)
    {}
    ~CudnnConvolutional2DComponent()
    {
        initialized_ = false;
      if(initialized_){
        CU_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc_));
        CU_SAFE_CALL(cudnnDestroyTensorDescriptor(out_desc_));
        CU_SAFE_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
        CU_SAFE_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
        CU_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
        cudaStreamDestroy(stream_);
        cudnnDestroy(handle_);
      }
    }
    
    Component* Copy() const {return new CudnnConvolutional2DComponent(*this);}
    ComponentType GetType() const {return kCudnnConvolutional2DComponent;}
    
    void InitData(std::istream &is){
        BaseFloat bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
        BaseFloat learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
        std::string token;
        while(!is.eof()){
            ReadToken(is, false, &token);
            /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
            else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
            else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
            else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
            else if (token == "<FmapXLen>")    ReadBasicType(is, false, &fmap_x_len_);
            else if (token == "<FmapYLen>")    ReadBasicType(is, false, &fmap_y_len_);
            else if (token == "<FiltXLen>")    ReadBasicType(is, false, &filt_x_len_);
            else if (token == "<FiltYLen>")    ReadBasicType(is, false, &filt_y_len_);
            else if (token == "<FiltXStep>")   ReadBasicType(is, false, &filt_x_step_);
            else if (token == "<FiltYStep>")   ReadBasicType(is, false, &filt_y_step_);
            else if (token == "<PadXLen>")  ReadBasicType(is, false, &pad_x_len_);
            else if (token == "<PadYLen>") ReadBasicType(is, false, &pad_y_len_);
            else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
            else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|FmapXLen|FmapYLen|FiltXLen|FiltYLen|FiltXStep|FiltYStep|PadXLen|PadYLen|LearnRateCoef|BiasLearnRateCoef)";
            is >> std::ws;  // eat-up whitespace
        }
        KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
        num_input_fmaps_ = input_dim_ / (fmap_x_len_ * fmap_y_len_);
        KALDI_LOG << "num_input_fmaps " << num_input_fmaps_;

        KALDI_ASSERT((fmap_x_len_ + 2 * pad_x_len_ - filt_x_len_) % (filt_x_step_) == 0);
        KALDI_ASSERT((fmap_y_len_ + 2 * pad_y_len_ - filt_y_len_) % (filt_y_step_) == 0);
        out_fmap_x_len_ = (fmap_x_len_ + 2 * pad_x_len_ - filt_x_len_)/filt_x_step_ + 1;
        out_fmap_y_len_ = (fmap_y_len_ + 2 * pad_y_len_ - filt_y_len_)/filt_y_step_ + 1;

        KALDI_ASSERT(output_dim_ % (out_fmap_x_len_ * out_fmap_y_len_)  == 0);
        num_output_fmaps_ = output_dim_ / (out_fmap_x_len_ * out_fmap_y_len_);
        KALDI_LOG << "num_output_fmaps " << num_output_fmaps_;
        int32 num_filters = output_dim_/(out_fmap_x_len_ * out_fmap_y_len_);
        KALDI_LOG << "num_filters " << num_filters;
        Matrix<BaseFloat> mat(num_filters, num_input_fmaps_ * filt_x_len_*filt_y_len_);
        //KALDI_LOG <<__FILE__<<__LINE__<<" "<<mat.NumRows()<<" "<<mat.NumCols(); 
        for (int32 r = 0; r < num_filters; r++) {
            for (int32 c = 0; c < num_input_fmaps_*filt_x_len_*filt_y_len_; c++) {
                // 0-mean Gauss with given std_dev
                if (param_range == 0.0)
                    mat(r, c) = param_stddev * RandGauss();
                else
                    mat(r,c) = param_range * (RandUniform() - 0.5) * 2;
            }
        }
        //KALDI_LOG <<__FILE__<<__LINE__<<mat.NumRows()<<" "<<mat.NumCols();
        filters_ = mat;
        Vector<BaseFloat> vec(num_filters);
        for (int32 i = 0; i < num_filters; i++) {
            vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range;
        }
        bias_ = vec;
        learn_rate_coef_ = learn_rate_coef;
        bias_learn_rate_coef_ = bias_learn_rate_coef;
        

    }
    
    void ReadData(std::istream &is, bool binary){
       
            ExpectToken(is, binary, "<LearnRateCoef>");
            ReadBasicType(is, binary, &learn_rate_coef_);
            ExpectToken(is, binary, "<BiasLearnRateCoef>");
            ReadBasicType(is, binary, &bias_learn_rate_coef_);
            ExpectToken(is, binary, "<FmapXLen>");
            ReadBasicType(is, binary, &fmap_x_len_);
            ExpectToken(is, binary, "<FmapYLen>");
            ReadBasicType(is, binary, &fmap_y_len_);
            ExpectToken(is, binary, "<FiltXLen>");
            ReadBasicType(is, binary, &filt_x_len_);
            ExpectToken(is, binary, "<FiltYLen>");
            ReadBasicType(is, binary, &filt_y_len_);
            ExpectToken(is, binary, "<FiltXStep>");
            ReadBasicType(is, binary, &filt_x_step_);
            ExpectToken(is, binary, "<FiltYStep>");
            ReadBasicType(is, binary, &filt_y_step_);
            ExpectToken(is, binary, "<PadXLen>");
            ReadBasicType(is, binary, &pad_x_len_);
            ExpectToken(is, binary, "<PadYLen>");
            ReadBasicType(is, binary, &pad_y_len_);
            ExpectToken(is, binary, "<Filters>");
            CuMatrix<BaseFloat> filters; 
            filters.Read(is, binary);
            filters_.Resize(filters.NumRows(), filters.NumCols(), kUndefined, kStrideEqualNumCols);
            filters_.CopyFromMat(filters);
            ExpectToken(is, binary, "<Bias>");
            bias_.Read(is, binary);
            KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
            KALDI_ASSERT((fmap_x_len_ + 2 * pad_x_len_ - filt_x_len_) % (filt_x_step_) == 0);
            KALDI_ASSERT((fmap_y_len_ + 2 * pad_y_len_ - filt_y_len_) % (filt_y_step_) == 0);
            out_fmap_x_len_ = (fmap_x_len_ + 2 * pad_x_len_ - filt_x_len_)/filt_x_step_ + 1;
            out_fmap_y_len_ = (fmap_y_len_ + 2 * pad_y_len_ - filt_y_len_)/filt_y_step_ + 1;
            KALDI_ASSERT(output_dim_ % (out_fmap_x_len_ * out_fmap_y_len_)  == 0);
            num_output_fmaps_ = output_dim_ / (out_fmap_x_len_ * out_fmap_y_len_);
            num_input_fmaps_ = input_dim_ / (fmap_x_len_ * fmap_y_len_);
            filters_grad_.Resize(filters_.NumRows(), filters_.NumCols(), kSetZero, kStrideEqualNumCols);
            bias_grad_.Resize(filters_.NumRows());
            
    }

    void WriteData(std::ostream &os, bool binary) const{
            WriteToken(os, binary, "<LearnRateCoef>");
            WriteBasicType(os, binary, learn_rate_coef_);
            WriteToken(os, binary, "<BiasLearnRateCoef>");
            WriteBasicType(os, binary, bias_learn_rate_coef_);
            WriteToken(os, binary, "<FmapXLen>");
            WriteBasicType(os, binary, fmap_x_len_);
            WriteToken(os, binary, "<FmapYLen>");
            WriteBasicType(os, binary, fmap_y_len_);
            WriteToken(os, binary, "<FiltXLen>");
            WriteBasicType(os, binary, filt_x_len_);
            WriteToken(os, binary, "<FiltYLen>");
            WriteBasicType(os, binary, filt_y_len_);
            WriteToken(os, binary, "<FiltXStep>");
            WriteBasicType(os, binary, filt_x_step_);
            WriteToken(os, binary, "<FiltYStep>");
            WriteBasicType(os, binary, filt_y_step_);
            WriteToken(os, binary, "<PadXLen>");
            WriteBasicType(os, binary, pad_x_len_);
            WriteToken(os, binary, "<PadYLen>");
            WriteBasicType(os, binary, pad_y_len_);
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
        Matrix<BaseFloat> filters_tmp(filters_.NumRows(), filters_.NumCols());
        filters_tmp.CopyFromMat(filters_); 
        wei_copy->Range(0, filters_num_elem).CopyRowsFromMat(filters_tmp);
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
    
    inline void Init(){
        
        //size_t workspace_byte = 8*1024*1024 ;
        cudaStreamCreate(&stream_);
        cudnnCreate(&handle_);
        cudnnSetStream(handle_, stream_);
      
        CU_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc_));
        CU_SAFE_CALL(cudnnCreateTensorDescriptor(&out_desc_));
        CU_SAFE_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
        CU_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
        CU_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    }

    void ReShape(int batch_size){ 
        
       size_t back_size = 0 ;
       size_t back_size_w = 0 ;
       cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW ; 
       CU_SAFE_CALL(cudnnSetFilter4dDescriptor(filter_desc_, 
                                            CUDNN_DATA_FLOAT, 
                                            format, 
                                            num_output_fmaps_, 
                                            num_input_fmaps_, 
                                            filt_y_len_, 
                                            filt_x_len_));
       CU_SAFE_CALL(cudnnSetConvolution2dDescriptor(conv_desc_, 
                                                 pad_y_len_, 
                                                 pad_x_len_, 
                                                 filt_y_step_, 
                                                 filt_x_step_, 
                                                 1,
                                                 1, 
                                                 CUDNN_CONVOLUTION));

       CU_SAFE_CALL(cudnnSetTensor4dDescriptorEx(in_desc_, 
                                              CUDNN_DATA_FLOAT, 
                                              batch_size, 
                                              num_input_fmaps_, 
                                              fmap_y_len_, 
                                              fmap_x_len_, 
                                              fmap_y_len_ * fmap_x_len_ * num_input_fmaps_,
                                              fmap_y_len_ * fmap_x_len_,
                                              fmap_x_len_, 1));
        CU_SAFE_CALL(cudnnSetTensor4dDescriptorEx(out_desc_,
                                              CUDNN_DATA_FLOAT,
                                              batch_size,
                                              num_output_fmaps_,
                                              out_fmap_y_len_ ,
                                              out_fmap_x_len_ ,
                                              out_fmap_y_len_ * out_fmap_x_len_ * num_output_fmaps_,
                                              out_fmap_y_len_ * out_fmap_x_len_,
                                              out_fmap_x_len_,
                                              1));

        int32 bias_offset = num_output_fmaps_;
        std::vector<int> bias_shape ;
        bias_shape.push_back(1);
        bias_shape.push_back(bias_offset);
        bias_shape.push_back(1);
        bias_shape.push_back(1);
        std::vector<int> bias_stride ;
        bias_stride.push_back(bias_offset);
        bias_stride.push_back(1);
        bias_stride.push_back(1);
        bias_stride.push_back(1);
        CU_SAFE_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                            CUDNN_DATA_FLOAT,
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]));

      
        //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        CU_SAFE_CALL(cudnnGetConvolutionForwardAlgorithm(handle_,
                 in_desc_,
                 filter_desc_,
                 conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                 0,
                 &algo_));

        CU_SAFE_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
                 in_desc_,
                 out_desc_,
                 conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                 0,
                 &back_algo_w_));

        CU_SAFE_CALL(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
                 filter_desc_,
                 out_desc_,
                 conv_desc_,
                 in_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                 0,
                 &back_algo_));

        CU_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
                filter_desc_,
                out_desc_,
                conv_desc_,
                in_desc_,
                back_algo_,
                &back_size));

        CU_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
                in_desc_,
                out_desc_,
                conv_desc_,
                filter_desc_,
                back_algo_w_,
                &back_size_w));


        backward_workspace_byte_ = std::max(back_size, back_size_w);

        CU_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle_,
                in_desc_,
                filter_desc_,
                conv_desc_,
                out_desc_,
                algo_,
                &forward_workspace_byte_));  // get workspace bytes

        BaseFloat forward_workspace_size_ = forward_workspace_byte_ / sizeof(float) + 1;
        BaseFloat backward_workspace_size_ = backward_workspace_byte_ / sizeof(float) + 1;
        
        forward_workspace_.Resize(forward_workspace_size_, kUndefined);
        backward_workspace_.Resize(backward_workspace_size_, kUndefined);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        if(!initialized_){
            Init();
            initialized_ = true ;
            batch_size_ = in.NumRows();
            ReShape(batch_size_);
        }
        if(batch_size_ != in.NumRows()){
            batch_size_ = in.NumRows();
            ReShape(batch_size_);
        }
        BaseFloat alpha = 1.0f ;
        BaseFloat beta = 0.0f ;
        const BaseFloat *in_ptr = in.Data() ;
        BaseFloat *filters_ptr = filters_.Data();
        BaseFloat *out_ptr = out->Data();
        BaseFloat* bias_ptr = bias_.Data();
        BaseFloat* forward_workspace_ptr = forward_workspace_.Data();

        CU_SAFE_CALL(cudnnConvolutionForward(handle_,
                                         &alpha,
                                         in_desc_,
                                         in_ptr,
                                         filter_desc_,
                                         filters_ptr,
                                         conv_desc_,
                                         algo_,
                                         forward_workspace_ptr,
                                         forward_workspace_byte_,
                                         &beta,
                                         out_desc_,
                                         out_ptr));
        beta = 1.0f;
        CU_SAFE_CALL(cudnnAddTensor(handle_,
                                &alpha,
                                bias_desc_,
                                bias_ptr,
                                &beta,
                                out_desc_,
                                out_ptr));
    }
    
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        BaseFloat alpha = 1.0f;
        BaseFloat beta = 0.0f;
        const BaseFloat* out_diff_ptr = out_diff.Data() ;
        BaseFloat* in_diff_ptr = in_diff->Data() ;
        BaseFloat* filters_ptr = filters_.Data();
        BaseFloat* backward_workspace_ptr = backward_workspace_.Data();

        CU_SAFE_CALL(cudnnConvolutionBackwardData(handle_,
                                              &alpha,
                                              filter_desc_,
                                              filters_ptr,
                                              out_desc_,
                                              out_diff_ptr,
                                              conv_desc_,
                                              back_algo_,
                                              backward_workspace_ptr,
                                              backward_workspace_byte_,
                                              &beta,
                                              in_desc_,
                                              in_diff_ptr));
    }   

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {

        Gradient(input, diff);

        BaseFloat lr = opts_.learn_rate ;
        //filters_grad_.Scale(1.0/(out_fmap_x_len_ * out_fmap_y_len_));
        bias_grad_.Scale(1.0/(out_fmap_x_len_ * out_fmap_y_len_));
        filters_.AddMat(-lr*learn_rate_coef_, filters_grad_);
        bias_.AddVec(-lr*bias_learn_rate_coef_, bias_grad_);
    }

    void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        BaseFloat mmt = opts_.momentum;
        BaseFloat alpha = 1.0f;
        BaseFloat beta = mmt;
        const BaseFloat* diff_ptr = diff.Data() ;
        BaseFloat* bias_grad_ptr = bias_grad_.Data();
        const BaseFloat* input_ptr = input.Data();
        BaseFloat* filters_grad_ptr = filters_grad_.Data();
        BaseFloat* backward_workspace_ptr = backward_workspace_.Data();

        CU_SAFE_CALL(cudnnConvolutionBackwardBias(handle_,
                                              &alpha,
                                              out_desc_,
                                              diff_ptr,
                                              &beta,
                                              bias_desc_,
                                              bias_grad_ptr));

                 cudnnConvolutionBackwardFilter(handle_,
                                                &alpha,
                                                in_desc_,
                                                input_ptr,
                                                out_desc_,
                                                diff_ptr,
                                                conv_desc_,
                                                back_algo_w_,
                                                backward_workspace_ptr,
                                                backward_workspace_byte_,
                                                &beta,
                                                filter_desc_,
                                                filters_grad_ptr);
  }

    void UpdateGradient(){
        const BaseFloat lr = opts_.learn_rate;
        //filters_grad_.Scale(1.0/(out_fmap_x_len_ * out_fmap_y_len_));
        bias_grad_.Scale(1.0/(out_fmap_x_len_ * out_fmap_y_len_));
        filters_.AddMat(-lr*learn_rate_coef_, filters_grad_);
        bias_.AddVec(-lr*bias_learn_rate_coef_, bias_grad_);
    }    

    void ResetGradient()
    {
        filters_grad_.SetZero();
        bias_grad_.SetZero();
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
      int32 fmap_x_len_, fmap_y_len_,
        filt_x_len_, filt_y_len_,
        filt_x_step_, filt_y_step_,
        pad_x_len_, pad_y_len_,
        num_output_fmaps_,num_input_fmaps_,
        out_fmap_x_len_, out_fmap_y_len_;
      BaseFloat learn_rate_coef_;
      BaseFloat bias_learn_rate_coef_;
      CuMatrix<BaseFloat> filters_;  ///< row = vectorized rectangular filter
      CuVector<BaseFloat> bias_;  ///< bias for each filter

      CuMatrix<BaseFloat> filters_grad_;  ///< gradient of filters
      CuVector<BaseFloat> bias_grad_;  ///< gradient of biases
      bool initialized_ ;
      size_t batch_size_ ;
      CuVector<BaseFloat> forward_workspace_;
      CuVector<BaseFloat> backward_workspace_;
      size_t forward_workspace_byte_ ;
      size_t backward_workspace_byte_;
      cudnnTensorDescriptor_t in_desc_ ;
      cudnnTensorDescriptor_t out_desc_ ;
      cudnnTensorDescriptor_t bias_desc_ ;
      cudnnFilterDescriptor_t filter_desc_ ;
      cudnnConvolutionDescriptor_t conv_desc_ ;
      cudnnConvolutionFwdAlgo_t algo_ ;
      cudnnConvolutionBwdDataAlgo_t back_algo_ ;
      cudnnConvolutionBwdFilterAlgo_t back_algo_w_ ;
      cudnnHandle_t handle_ ;
      cudaStream_t stream_ ;

};

}//end of namespace nnet1
}//end of namespace kaldi
#endif

#endif //NNET_CUDNN_CONVOLUTIONAL_2D_COMPONENT_H_
