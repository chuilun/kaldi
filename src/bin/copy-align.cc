// featbin/copy-align.cc

// Copyright 2009-2011  Microsoft Corporation
//           2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy alignments [and possibly change format]\n"
        "Usage: copy-align [options] <feature-rspecifier> <feature-wspecifier>\n"
        "or:   copy-align [options] <feats-rxfilename> <feats-wxfilename>\n"
        "e.g.: copy-align ark:- ark,scp:foo.ark,foo.scp\n"
        " or: copy-align ark:foo.ark ark,t:txt.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);

      SequentialInt32VectorReader num_ali_reader(rspecifier);
      Int32VectorWriter alignment_writer(wspecifier);

      for (; !num_ali_reader.Done(); num_ali_reader.Next(), num_done++)
    	  alignment_writer.Write(num_ali_reader.Key(), num_ali_reader.Value());

      KALDI_LOG << "Copied " << num_done << " alignment utterances.";
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


