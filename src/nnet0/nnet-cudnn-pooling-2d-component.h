#ifndef NNET_CUDNN_POOLING_2D_COMPONENT_H_
#define NNET_CUDNN_POOLING_2D_COMPONENT_H_

#if HAVE_CUDA == 1
#include <cudnn.h>

#include "nnet/nnet-component.h"

namespace kaldi{
namespace nnet0{
    class CudnnPooling2DComponent : public Component{
    public:

        CudnnPooling2DComponent(int32 dim_in, int32 dim_out):Component(dim_in, dim_out),fmap_x_len_(0),fmap_y_len_(0),pool_x_len_(0),pool_y_len_(0),
        pool_x_step_(0), pool_y_step_(0),pad_x_len_(0),pad_y_len_(0),dtype_(CUDNN_DATA_FLOAT), 
        mode_(CUDNN_POOLING_MAX), initialized_(false)
        {}
        ~CudnnPooling2DComponent()
        {
            initialized_ = false;
            if(initialized_){
                CU_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc_));
                CU_SAFE_CALL(cudnnDestroyTensorDescriptor(out_desc_));
                CU_SAFE_CALL(cudnnDestroyPoolingDescriptor(pooling_desc_));
                cudaStreamDestroy(stream_);
                cudnnDestroy(handle_);
            }
        }

        ComponentType GetType() const{
            return kCudnnPooling2DComponent;
        }

        Component* Copy() const { return new CudnnPooling2DComponent(*this); }

        void InitData(std::istream &is){

            std::string token;
            while(!is.eof()){
                ReadToken(is, false, &token);
                 if(token == "<FmapXLen>") ReadBasicType(is, false, &fmap_x_len_);
                else if(token == "<FmapYLen>") ReadBasicType(is, false,  &fmap_y_len_);
                else if(token == "<PoolXLen>") ReadBasicType(is, false, &pool_x_len_);
                else if(token == "<PoolYLen>") ReadBasicType(is, false, &pool_y_len_);
                else if(token == "<PoolXStep>") ReadBasicType(is, false, &pool_x_step_);
                else if(token == "<PoolYStep>") ReadBasicType(is, false, &pool_y_step_);
                else if(token == "<PadXLen>") ReadBasicType(is, false, &pad_x_len_);
                else if(token == "<PadYLen>") ReadBasicType(is, false, &pad_y_len_);
                else KALDI_ERR<<"Unknown Token "<<token << ", a typo in config? "
                              << "(FmapXLen|FmapYLen|PoolXStep|PoolYStep|PoolXLen|PoolYLen|PadXLen|PadYLen)";
                is >> std::ws ; 
            }
            KALDI_ASSERT((fmap_x_len_ + 2*pad_x_len_ - pool_x_len_)%pool_x_step_ == 0);
            KALDI_ASSERT((fmap_y_len_ + 2*pad_y_len_ - pool_y_len_)%pool_y_step_ == 0);
            KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_)==0) ;
            num_input_fmaps_ = input_dim_ /(fmap_x_len_ * fmap_y_len_);
        }

        void Init(int32 batch_size){

            cudaStreamCreate(&stream_);
            cudnnCreate(&handle_);
            cudnnSetStream(handle_, stream_);
            nan_prop_ = CUDNN_NOT_PROPAGATE_NAN ;
            
            CU_SAFE_CALL(cudnnCreatePoolingDescriptor(&pooling_desc_));
            CU_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc_));
            CU_SAFE_CALL(cudnnCreateTensorDescriptor(&out_desc_));
         }

         void ReShape(int batch_size){

            CU_SAFE_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                                                CUDNN_TENSOR_NCHW,
                                                dtype_,
                                                batch_size,
                                                num_input_fmaps_,
                                                fmap_y_len_,
                                                fmap_x_len_));

            CU_SAFE_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                                CUDNN_TENSOR_NCHW,
                                                dtype_,
                                                batch_size,
                                                num_input_fmaps_,
                                                out_fmap_y_len_,
                                                out_fmap_x_len_));


            CU_SAFE_CALL(cudnnSetPooling2dDescriptor(pooling_desc_,
                                                 CUDNN_POOLING_MAX,
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 pool_y_len_,
                                                 pool_x_len_,
                                                 pad_y_len_,
                                                 pad_x_len_,
                                                 pool_y_len_,
                                                 pool_x_len_));
        }


        void ReadData(std::istream &is, bool binary){

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
            ExpectToken(is, binary, "<PadXLen>");
            ReadBasicType(is, binary, &pad_x_len_);
            ExpectToken(is, binary,"<PadYLen>");
            ReadBasicType(is, binary, &pad_y_len_);

            assert((fmap_x_len_ + 2*pad_x_len_ - pool_x_len_)%pool_x_step_ == 0);
            assert((fmap_y_len_ + 2*pad_y_len_ - pool_y_len_)%pool_y_step_ == 0);

            out_fmap_x_len_ = (fmap_x_len_ + 2*pad_x_len_ - pool_x_len_)/pool_x_step_ + 1 ;
            out_fmap_y_len_ = (fmap_y_len_ + 2*pad_y_len_ - pool_y_len_)/pool_y_step_ + 1 ;
            KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_)==0) ;
            num_input_fmaps_ = input_dim_ /(fmap_x_len_ * fmap_y_len_);

        }

        void WriteData(std::ostream &os, bool binary)const {
            WriteToken(os, binary, "<FmapXLen>") ;
            WriteBasicType(os, binary, fmap_x_len_);
            WriteToken(os, binary, "<FmapYLen>") ;
            WriteBasicType(os, binary, fmap_y_len_);
            WriteToken(os, binary, "<PoolXLen>") ;
            WriteBasicType(os, binary, pool_x_len_);
            WriteToken(os, binary, "<PoolYLen>") ;
            WriteBasicType(os,binary, pool_y_len_);
            WriteToken(os,binary, "<PoolXStep>");
            WriteBasicType(os, binary, pool_x_step_);
            WriteToken(os, binary, "<PoolYStep>");
            WriteBasicType(os,binary, pool_y_step_);
            WriteToken(os,binary, "<PadXLen>") ;
            WriteBasicType(os,binary, pad_x_len_);
            WriteToken(os,binary, "<PadYLen>") ;
            WriteBasicType(os,binary, pad_y_len_);
        }
    
        void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
            
            if(!initialized_){
                Init(in.NumRows()) ;
                initialized_ = true;
                batch_size_ = in.NumRows();
                ReShape(batch_size_);
            }
            
            if(batch_size_ != in.NumRows()){
                batch_size_ = in.NumRows();
                ReShape(batch_size_);
            }

            BaseFloat alpha = 1.0f, beta = 0.0f ;
            const BaseFloat* data_ptr = in.Data() ;
            BaseFloat* out_ptr = out->Data() ;
            CU_SAFE_CALL(cudnnPoolingForward(handle_,
                                         pooling_desc_,
                                         &alpha,
                                         in_desc_,
                                         data_ptr,
                                         &beta,
                                         out_desc_,
                                         out_ptr));
        }

        void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                      const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

            BaseFloat alpha = 1.0f, beta = 0.0f ;
            const BaseFloat* out_dptr = out.Data();
            const BaseFloat* out_diff_dptr = out_diff.Data();
            const BaseFloat* in_ptr = in.Data();
            BaseFloat* in_diff_ptr = in_diff->Data();
            CU_SAFE_CALL(cudnnPoolingBackward(handle_,
                                          pooling_desc_,
                                          &alpha,
                                          out_desc_,
                                          out_dptr,
                                          out_desc_,
                                          out_diff_dptr,
                                          in_desc_,
                                          in_ptr,
                                          &beta,
                                          in_desc_,
                                          in_diff_ptr));
        }

        private:
            int32 fmap_x_len_, fmap_y_len_, out_fmap_x_len_, out_fmap_y_len_, pool_x_len_,pool_y_len_,
            pool_x_step_, pool_y_step_, pad_x_len_, pad_y_len_, num_input_fmaps_; 
            cudnnDataType_t dtype_;
            cudnnHandle_t handle_;
            cudaStream_t stream_ ;
            cudnnPoolingMode_t mode_;
            cudnnTensorDescriptor_t in_desc_;
            cudnnTensorDescriptor_t out_desc_;
            cudnnPoolingDescriptor_t pooling_desc_;
            cudnnNanPropagation_t nan_prop_;
            bool initialized_ ;
            size_t batch_size_ ;
    };


}
}
#endif
#endif
