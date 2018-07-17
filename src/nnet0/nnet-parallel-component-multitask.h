// nnet0/nnet-parallel-component-multitask.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)
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


#ifndef KALDI_NNET_NNET_PARALLEL_COMPONENT_MULTITASK_H_
#define KALDI_NNET_NNET_PARALLEL_COMPONENT_MULTITASK_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"

#include <sstream>

namespace kaldi {
namespace nnet0 {

class ParallelComponentMultiTask : public UpdatableComponent {
 public:
	ParallelComponentMultiTask(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out)
  { }
  ~ParallelComponentMultiTask()
  { }

  Component* Copy() const { return new ParallelComponentMultiTask(*this); }
  ComponentType GetType() const { return kParallelComponentMultiTask; }

  void InitData(std::istream &is) {
    // define options
    // std::vector<std::string> nested_nnet_proto;
    // std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token, name;
	int32 offset, len = 0; 
    BaseFloat scale, escale;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      if (token == "<NestedNnet>" || token == "<NestedNnetFilename>") {
    	  ExpectToken(is, false, "<Name>");
    	  ReadToken(is, false, &name);

		  ExpectToken(is, false, "<InputOffset>");
		  ReadBasicType(is, false, &offset);
		  input_offset[name] = std::pair<int32, int32>(offset, len);

		  ExpectToken(is, false, "<OutputOffset>");
		  ReadBasicType(is, false, &offset);
		  output_offset[name] = std::pair<int32, int32>(offset, len);

		  ExpectToken(is, false, "<Scale>");
		  ReadBasicType(is, false, &scale);
		  forward_scale[name] = scale;

		  ExpectToken(is, false, "<ErrorScale>");
		  ReadBasicType(is, false, &escale);
		  error_scale[name] = escale;

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);

          // read nnets from files
          Nnet nnet;
          nnet.Read(file_or_end);
          nnet_[name] = nnet;
          input_offset[name].second = nnet.InputDim();
          output_offset[name].second = nnet.OutputDim();
          KALDI_LOG << "Loaded nested <Nnet> from file : " << file_or_end;

          ReadToken(is, false, &file_or_end);
          KALDI_ASSERT(file_or_end == "</NestedNnet>" || file_or_end == "</NestedNnetFilename>");

      } else if (token == "<NestedNnetProto>") {
    	  ExpectToken(is, false, "<Name>");
    	  ReadToken(is, false, &name);

      	  ExpectToken(is, false, "<InputOffset>");
      	  ReadBasicType(is, false, &offset);
      	  input_offset[name] = std::pair<int32, int32>(offset, len);

      	  ExpectToken(is, false, "<OutputOffset>");
      	  ReadBasicType(is, false, &offset);
      	  output_offset[name] = std::pair<int32, int32>(offset, len);

		  ExpectToken(is, false, "<Scale>");
		  ReadBasicType(is, false, &scale);
		  forward_scale[name] = scale;

		  ExpectToken(is, false, "<ErrorScale>");
		  ReadBasicType(is, false, &escale);
		  error_scale[name] = escale;

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);

          // initialize nnets from prototypes
          Nnet nnet;
          nnet.Init(file_or_end);
          nnet_[name] = nnet;
          input_offset[name].second = nnet.InputDim();
          output_offset[name].second = nnet.OutputDim();
          KALDI_LOG << "Initialized nested <Nnet> from prototype : " << file_or_end;

          ReadToken(is, false, &file_or_end);
          KALDI_ASSERT(file_or_end == "</NestedNnetProto>");

      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
                       << " (NestedNnet|NestedNnetFilename|NestedNnetProto)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    // KALDI_ASSERT((nested_nnet_proto.size() > 0) ^ (nested_nnet_filename.size() > 0)); //xor
    KALDI_ASSERT(nnet_.size() > 0);

    // check dim-sum of nested nnets
    check();
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    std::pair<int32, int32> offset;
    std::string name;
    int32 nnet_count;
    BaseFloat scale, escale;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);

      ExpectToken(is, false, "<Name>");
      ReadToken(is, false, &name);

      ExpectToken(is, binary, "<InputOffset>");
      ReadBasicType(is, binary, &offset.first);
      input_offset[name] = offset;

      ExpectToken(is, binary, "<OutputOffset>");
      ReadBasicType(is, binary, &offset.first);
      output_offset[name] = offset;

      ExpectToken(is, binary, "<Scale>");
      ReadBasicType(is, binary, &scale);
      forward_scale[name] = scale;

      ExpectToken(is, binary, "<ErrorScale>");
      ReadBasicType(is, binary, &escale);
      error_scale[name] = escale;

      Nnet nnet;
      nnet.Read(is, binary);
      nnet_[name] = nnet;
      input_offset[name].second = nnet.InputDim();
      output_offset[name].second = nnet.OutputDim();
    }
    ExpectToken(is, binary, "</ParallelComponentMultiTask>");

    // check dim-sum of nested nnets
    check();
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    std::string name;
    //unordered_map<std::string, std::pair<int32, int32> >::iterator it;
    int32 i = 0;
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);

    for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
    {
    	name = it->first;
        WriteToken(os, binary, "<NestedNnet>");
        WriteBasicType(os, binary, i+1);

        WriteToken(os, binary, "<Name>");
        WriteToken(os, binary, name);

        WriteToken(os, binary, "<InputOffset>");
        WriteBasicType(os, binary, input_offset.find(name)->second.first);

        WriteToken(os, binary, "<OutputOffset>");
        WriteBasicType(os, binary, output_offset.find(name)->second.first);

        WriteToken(os, binary, "<Scale>");
        WriteBasicType(os, binary, forward_scale.find(name)->second);

        WriteToken(os, binary, "<ErrorScale>");
        WriteBasicType(os, binary, error_scale.find(name)->second);

        if(binary == false) os << std::endl;
        nnet_.find(name)->second.Write(os, binary);
        if(binary == false) os << std::endl;
    }
    WriteToken(os, binary, "</ParallelComponentMultiTask>");
  }

  int32 NumParams() const { 
    int32 num_params_sum = 0;
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
      num_params_sum += it->second.NumParams();
    return num_params_sum;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const { 
    wei_copy->Resize(NumParams());
    int32 offset = 0;
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      Vector<BaseFloat> wei_aux;
      it->second.GetParams(&wei_aux);
      wei_copy->Range(offset, wei_aux.Dim()).CopyFromVec(wei_aux);
      offset += wei_aux.Dim();
    }
    KALDI_ASSERT(offset == NumParams());
  }
    
  std::string Info() const { 
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_network #" << it->first << "{\n" << it->second.Info() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }
                       
  std::string InfoGradient() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_gradient #" << it->first << "{\n" << it->second.InfoGradient() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_propagate #" << it->first << "{\n" << it->second.InfoPropagate() << "}\n";
    }
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_backpropagate #" << it->first << "{\n" <<  it->second.InfoBackPropagate() << "}\n";
    }
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {

	  std::string name;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  name = it->first;
		  CuSubMatrix<BaseFloat> src(in.ColRange(input_offset[name].first, input_offset[name].second));
		  CuSubMatrix<BaseFloat> tgt(out->ColRange(output_offset[name].first, output_offset[name].second));
		  //
		  CuMatrix<BaseFloat> tgt_aux;
		  nnet_[name].Propagate(src, &tgt_aux);
		  tgt.AddMat(forward_scale[name], tgt_aux);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
	  std::string name;
	  in_diff->SetZero();
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  name = it->first;
		  CuSubMatrix<BaseFloat> src(out_diff.ColRange(output_offset[name].first, output_offset[name].second));
		  CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(input_offset[name].first, input_offset[name].second));
		  //
		  CuMatrix<BaseFloat> tgt_aux;
		  nnet_[name].Backpropagate(src, &tgt_aux, false);
		  tgt.AddMat(error_scale[name], tgt_aux);
    }
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
      for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
          it->second.Gradient();
  }

  void UpdateGradient() {
      for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
          it->second.UpdateGradient();
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    ; // do nothing
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  it->second.SetTrainOptions(opts);
    }
  }

  int32 GetDim() const
  {
	  int32 dim = 0;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
		  dim += it->second.GetDim();
	  return dim;
  }

  int WeightCopy(void *host, int direction, int copykind)
  {
	  int pos = 0;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
		  pos += it->second.WeightCopy((void*)((char *)host+pos), direction, copykind);
	  return pos;
  }

  Component* GetComponent(Component::ComponentType type)
  {
	  Component *com = NULL;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
	  {
		  Nnet &nnet = it->second;
		  for (int32 c = 0; c < nnet.NumComponents(); c++)
		  {

			  if (nnet.GetComponent(c).GetType() == type)
			  {
				  com = &nnet.GetComponent(c);
				  return com;
			  }
			  else if (nnet.GetComponent(c).GetType() == Component::kParallelComponentMultiTask)
			  {
				  com = (dynamic_cast<ParallelComponentMultiTask&>(nnet.GetComponent(c))).GetComponent(type);
				  if (com != NULL) return com;
			  }
		  }
	  }
	  return com;
  }

  std::unordered_map<std::string, std::pair<int32, int32> >	GetOutputOffset()
  {
	  return output_offset;
  }

  void SetErrorScale(std::unordered_map<std::string, BaseFloat> error_scale)
  {
      this->error_scale = error_scale;
  }

  void UpdateLstmStreamsState(const std::vector<int32> &update_state_flag)
  {
	  if (nnet_.find("lstm") != nnet_.end())
		  nnet_["lstm"].UpdateLstmStreamsState(update_state_flag);
  }

  void ResetLstmStreams(const std::vector<int32> &stream_reset_flag, int32 ntruncated_bptt_size)
  {
	  if (nnet_.find("lstm") != nnet_.end())
		  nnet_["lstm"].ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
  }

  std::unordered_map<std::string, Nnet> &GetNnet() {
	  return nnet_;
  }

 private:
  void check()
  {
	    // check dim-sum of nested nnets
	    int32 nnet_input_max = 0, nnet_output_max = 0, dim = 0;
	    std::string name;
	    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
	    	name = it->first;
	    	dim = input_offset[name].first + input_offset[name].second;
	    	if (nnet_input_max < dim) nnet_input_max = dim;

	    	dim = output_offset[name].first + output_offset[name].second;
	    	if (nnet_output_max < dim) nnet_output_max = dim;
	    }
	    KALDI_ASSERT(InputDim() >= nnet_input_max);
	    KALDI_ASSERT(OutputDim() == nnet_output_max);
  }

 private:
  std::unordered_map<std::string, Nnet> nnet_;
  std::unordered_map<std::string, std::pair<int32, int32> > input_offset;  // pair<offset, length>
  std::unordered_map<std::string, std::pair<int32, int32> > output_offset;
  std::unordered_map<std::string, BaseFloat> forward_scale;
  std::unordered_map<std::string, BaseFloat> error_scale;
};

} // namespace nnet0
} // namespace kaldi

#endif
