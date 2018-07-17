// nnetbin/nnet-extract.cc

// Copyright 2012-2013  Brno University of Technology (Author: Karel Vesely)
// Copyright 2016-2017  AISpeech (Author: Wei Deng)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-parallel-component-multitask.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    const char *usage =
        "Extract Neural Networks with specified output name in multitask task(and possibly change binary/text format)\n"
        "Usage:  nnet-extract [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " nnet-extract --binary=false --output-name=mmi nnet.in nnet.out\n";
    
    ParseOptions po(usage);
    
    bool binary_write = true;
    std::string output_name = "mmi";
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("output-name", &output_name, "last multitask output name");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);
    std::string model_out_filename = po.GetArg(2);

    //read the first nnet
    KALDI_LOG << "Reading " << model_in_filename;
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    ParallelComponentMultiTask *parallel = NULL;
    for (int32 c = 0; c < nnet.NumComponents(); c++) {   
        if (nnet.GetComponent(c).GetType() == Component::kParallelComponentMultiTask)
            parallel = &(dynamic_cast<ParallelComponentMultiTask&>(nnet.GetComponent(c)));
    }

    Nnet nnet_out = nnet;
    if (parallel != NULL) {
        std::unordered_map<std::string, Nnet> &mnet = parallel->GetNnet();
        if(mnet.find(output_name) == mnet.end())
            KALDI_ERR << "Can not find ouput network name " << output_name;
        nnet_out.RemoveLastComponent();
        nnet_out.AppendNnet(mnet[output_name]);
    }

    //finally write the nnet to disk
    {
      Output ko(model_out_filename, binary_write);
      nnet_out.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


