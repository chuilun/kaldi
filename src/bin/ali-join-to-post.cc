// bin/ali-join-to-post.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Join alignments [and possibly change format] to posterior format.\n"
        "Usage: ali-join <align-in-rspecifier1> [<align-in-rspecifier2>"
        " <align-in-rspecifier3> ...] <join-post-out-wspecifier>\n"
    	"  e.g.: ali-join-to-post --ali-dim=\"10092:411\" ark:ali1.ark ark:ali2.ark ark:join_post.ark\n";

    ParseOptions po(usage);

    std::string ali_dim_str;
    po.Register("ali-dim", &ali_dim_str, "Each input alignment dim.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_args = po.NumArgs();
    std::string ali_in_fn1 = po.GetArg(1),
    		posteriors_wspecifier = po.GetOptArg(num_args);

    // Output join alignment
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    // Input alignment
    SequentialInt32VectorReader ali_reader1(ali_in_fn1);
    std::vector<RandomAccessInt32VectorReader*> ali_readers(num_args-2,
                       static_cast<RandomAccessInt32VectorReader*>(NULL));

    std::vector<std::string> ali_in_fns(num_args-2);
    for (int32 i = 2; i < num_args; ++i) {
    	ali_readers[i-2] = new RandomAccessInt32VectorReader(po.GetArg(i));
        ali_in_fns[i-2] = po.GetArg(i);
    }

    std::vector<int32> ali_dim(num_args-1);
    if (!kaldi::SplitStringToIntegers(ali_dim_str, ":", false, &ali_dim))
        KALDI_ERR << "Invalid alignment dim string " << ali_dim_str;

    std::vector<int32> ali_dim_offset(num_args-1, 0);
    for (int32 i = 1; i < num_args-1; i++)
        ali_dim_offset[i] = ali_dim_offset[i-1] + ali_dim[i-1];

    int32 num_done = 0, num_missing = 0, num_mismatch = 0, i;
    Posterior post;
    for (; !ali_reader1.Done(); ali_reader1.Next()) {
        std::string key = ali_reader1.Key();
        std::vector<int32> ali1 = ali_reader1.Value();
        //ali_reader1.FreeCurrent();
        post.clear();
        post.resize(ali1.size());

        for (int32 j = 0; j < ali1.size(); j++) {
            std::pair<int32, BaseFloat>  dot(ali1[j] + ali_dim_offset[0], 1.0);
            post[j].push_back(dot);
        }

        for (i = 0; i < num_args-2; ++i) {
             if (ali_readers[i]->HasKey(key)) {
               std::vector<int32> ali2 = ali_readers[i]->Value(key);

               if (ali2.size() != ali1.size()) {
            	   KALDI_WARN << key << ", length mismatch of targets " << ali1.size() << ", " << ali2.size()
            			   << " for rspecifier: " << ali_in_fns[i];
                   num_mismatch++;
            	   break;
               }

               for (int32 j = 0; j < ali1.size(); j++) {
            	   std::pair<int32, BaseFloat>  dot(ali2[j] + ali_dim_offset[i+1], 1.0);
            	   post[j].push_back(dot);
               }

           } else {
                KALDI_WARN << "No alignment found for utterance " << key << " for "
                   << "system " << (i + 2) << ", rspecifier: "
                   << ali_in_fns[i];
                num_missing++;
                break;
           }
        }
        
        if (i < num_args-2) 
            continue;

        posterior_writer.Write(key, post);
        num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " alignment utterances successful, "
                << num_missing << " total missing utterances, "
                << num_mismatch << " total mismatch utterances.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


