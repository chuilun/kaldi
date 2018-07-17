// bin/copy-fst.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Copy raw fst to ark type.\n"
        "\n"
        "Usage:   copy-fst <fst-key> <fst-in> <graphs-wspecifier>\n"
        "e.g.: \n"
        " copy-fst fst-key raw.fst ark:key.fst\n";
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string key = po.GetArg(1);
    std::string fst_rxfilename = po.GetArg(2);
    std::string fst_wspecifier = po.GetArg(3);

    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *fst = fst::ReadFstKaldi(fst_rxfilename);

    TableWriter<fst::VectorFstHolder> fst_writer(fst_wspecifier);

    fst_writer.Write(key, *fst);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
