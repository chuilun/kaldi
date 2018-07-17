// nnetbin/nnet-copy.cc

// Copyright 2014-2015  Shanghai Jiao Tong University (author: Wei Deng)

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
#include "nnet0/nnet-affine-transform.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy Neural Network model (and possibly change binary/text format)\n"
        "Usage:  nnet-svd [options] <model-in> <model-out>\n"
        "e.g.:\n"
        " nnet-svd --binary=false --svd-threshold=0.55 nnet.in nnet_txt.out\n"
    	" nnet-svd --binary=false --min-rank=256 --svd-layers=1:0.55:0.55:0.55:0.45 nnet.in nnet_txt.out\n";


    bool binary_write = true;
    BaseFloat svd_threshold = 0.55;
    std::string svd_layers = "";
    int min_rank = 1, max_rank = 4096;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("svd-threshold", &svd_threshold, "Limit svd rank threshold");
    po.Register("min-rank", &min_rank, "Limit min svd rank");
    po.Register("max-rank", &max_rank, "Limit max svd rank");
    po.Register("svd-layers", &svd_layers, "Limit svd rank in each layer");

    po.Read(argc, argv);


    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load the network
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    AffineTransform *aff, *aa, *bb;
    Nnet nnet_out;

    int num_affine = 0;
    int32 i = 0, j = 0;
    for (i=0; i< nnet.NumComponents(); i++)
    {
    	aff = NULL;
    	aff = dynamic_cast<AffineTransform* >(&(nnet.GetComponent(i)));
    	if (aff != NULL)
    		num_affine++;
    }

    std::vector<BaseFloat> ranks(num_affine);

    if (svd_layers == "")
    {
    	ranks[0] = 1;
    	for (i = 1; i < num_affine-1; i++)
    		ranks[i] = svd_threshold;
    	ranks[num_affine-1] = svd_threshold - 0.1;
    }
    else
    {
    	i = 0;
    	const char *ss = svd_layers.c_str();
    	do{
        	sscanf(ss, "%f", &ranks[i]);
        	i++;
    	}while ((ss = strstr(ss, ":")) != NULL && ss++ && i < num_affine);

	for (; i < num_affine; i++)
		ranks[i] = svd_threshold;
    }

    j = 0;
    for (int32 i=0; i < nnet.NumComponents(); i++)
    {
    	aff = NULL;
    	aff = dynamic_cast<AffineTransform* >(&(nnet.GetComponent(i)));
    	if (aff != NULL)
    	{
    		//std::cout<<ranks[j]<<std::endl;
		if (ranks[j] < 1)	
		{
    			aff->LimitRank(ranks[j], min_rank, max_rank, &aa, &bb);
    			nnet_out.AppendComponent(aa);
    			nnet_out.AppendComponent(bb);
		}
		else
			nnet_out.AppendComponent(nnet.GetComponent(i).Copy());
		j++;
    	}
    	else
    	{
    		nnet_out.AppendComponent(nnet.GetComponent(i).Copy());
    	}
    }

    // store the network
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


