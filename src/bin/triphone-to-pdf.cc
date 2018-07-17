// bin/triphone-to-pdf.cc

// Copyright 2017-2018   Shanghai Jiao Tong University (author: Yongbin You)

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
#include "tree/context-dep.h"
#include "tree/build-tree-utils.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "transform triphone state to pdf-id.\n"
        "Usage:  triphone-to-pdf [options] <model-in> <tree> <triphone-pdf-id-out>\n"
        "e.g.: \n"
        " triphone-to-pdf 1.mdl tree 1.triphone_pdfid.map\n";

    int32 P = 1, N = 3;

    ParseOptions po(usage);
    po.Register("context-width", &N, "Context window size [must match build-tree]");
    po.Register("central-position", &P, "Central position in context window [must match build-tree]");

    std::string silence_phones_str;
    po.Register("silence-phones", &silence_phones_str, "Colon-separated list of integer id's of silence phones, e.g. 46:47");

    int32 pdf_class = 0;
    po.Register("pdf-class", &pdf_class, "Hmm state id, e.g. 0, 1 or 2");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
    	po.PrintUsage();
    	exit(1);
    }

    std::string model_filename = po.GetArg(1),
    	tree_filename = po.GetOptArg(2),
        pdf_map_wxfilename = po.GetOptArg(3);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      // There is more in this file but we don't need it.
    }
    ContextDependency ctx_dep;
    {
    	bool binary;
    	Input ki(tree_filename.c_str(), &binary);
    	ctx_dep.Read(ki.Stream(), binary);
    }

    KALDI_ASSERT(N==ctx_dep.ContextWidth());
    KALDI_ASSERT(P==ctx_dep.CentralPosition());

    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones)){
    	KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    }

    const std::vector<int32> &phones = trans_model.GetPhones();
    std::vector<int32> phones_del_sil;
    bool is_silphone;
    for(int i = 0; i < phones.size(); i++){
    	is_silphone = false;
    	for(int j = 0; j < silence_phones.size(); j++){
    		if(phones[i] == silence_phones[j]){
    			is_silphone = true;
    			break;
    		}
    	}
    	if (!is_silphone) phones_del_sil.push_back(phones[i]);
    }

    std::vector<std::vector<int32> > triphones_pdfid;
 	std::vector<int32> triphone(N);
 	int32 pdf_id;
    for(int i = 0; i < phones.size();i++){
    	triphone[0] = phones[i];
    	for(int j = 0; j < phones_del_sil.size(); j++){
    		triphone[1] = phones_del_sil[j];
    		for(int m = 0; m < phones.size(); m++){
    			triphone[2] = phones[m];
    		    if(!ctx_dep.Compute(triphone, pdf_class, &pdf_id)){
    		    	KALDI_ERR << "Something went wrong! " << triphone[0] << " " << triphone[1] << " " << triphone[2];
    		    } else {
    		    	triphones_pdfid.push_back(triphone);
    		    	triphones_pdfid[triphones_pdfid.size()-1].push_back(pdf_id);
    		    }
    		}
    	}
    }

    std::vector<int32> sil_phone(3,0);
    for(int j = 0; j < silence_phones.size(); j++){
    	sil_phone[1] = silence_phones[j];
    	if(!ctx_dep.Compute(sil_phone, 0, &pdf_id)){
    		KALDI_ERR << "Something went wrong! ";
    	} else {
    		sil_phone.push_back(pdf_id);
    		triphones_pdfid.push_back(sil_phone);
    	}
    }
    Output ko(pdf_map_wxfilename, false);
    for(int i = 0; i < triphones_pdfid.size(); i++){
    	for(int j = 0; j < triphones_pdfid[i].size();j++){
    		WriteBasicType(ko.Stream(), false, triphones_pdfid[i][j]);
    	}
    	ko.Stream() << '\n';
    }

    if (ko.Stream().fail()) {
    	KALDI_ERR << "Write triphone to pdf-id: write failed.";
    }

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
