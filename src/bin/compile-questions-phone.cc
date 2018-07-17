// bin/compile-questions.cc

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
#include "hmm/hmm-topology.h"
#include "tree/build-tree-questions.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compile questions\n"
        "Usage:  compile-questions [options] <questions-text-file> <questions-out>\n"
        "e.g.: \n"
        " compile-questions questions.txt questions.qst\n";
    bool binary = true;
    int32 P = 1, N = 3;
    int32 num_iters_refine = 0,
        leftmost_questions_truncate = -1;


    ParseOptions po(usage);
    po.Register("binary", &binary,
                "Write output in binary mode");
    po.Register("context-width", &N,
                "Context window size [must match acc-tree-stats].");
    po.Register("central-position", &P,
                "Central position in phone context window [must match acc-tree-stats]");
    po.Register("num-iters-refine", &num_iters_refine,
                "Number of iters of refining questions at each node.  >0 --> questions "
                "not refined");
    po.Register("leftmost-questions-truncate", &leftmost_questions_truncate,
                "If > 0, the questions for the left-most context position will be "
                "truncated to the specified number.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    // topo_filename = po.GetArg(1),
    std::string
        questions_rxfilename = po.GetArg(1),
        questions_out_filename = po.GetArg(2);

    // just needed for checking, and to get the
    // largest number of pdf-classes for any phone.

    // HmmTopology topo;
    // ReadKaldiObject(topo_filename, &topo);

    std::vector<std::vector<int32> > questions;  // sets of phones.
    if (!ReadIntegerVectorVectorSimple(questions_rxfilename, &questions))
      KALDI_ERR << "Could not read questions from "
                 << PrintableRxfilename(questions_rxfilename);
    for (size_t i = 0; i < questions.size(); i++) {
      std::sort(questions[i].begin(), questions[i].end());
      if (!IsSortedAndUniq(questions[i]))
        KALDI_ERR << "Questions contain duplicate phones";
    }
    size_t nq = static_cast<int32>(questions.size());
    SortAndUniq(&questions);
    if (questions.size() != nq)
      KALDI_WARN << (nq-questions.size())
                 << " duplicate questions present in " << questions_rxfilename;

    // ProcessTopo checks that all phones in the topo are
    // represented in at least one questions (else warns), and
    // returns the max # pdf classes in any given phone (normally
    // 3).
    // int32 max_num_pdfclasses = ProcessTopo(topo, questions);
    int32 max_num_pdfclasses = 1;
    Questions qo;

    QuestionsForKey phone_opts(num_iters_refine);
    // the questions-options corresponding to keys 0, 1, .. N-1 which
    // represent the phonetic context positions (including the central phone).
    for (int32 n = 0; n < N; n++) {
      KALDI_LOG << "Setting questions for phonetic-context position "<< n;
      if (n == 0 && leftmost_questions_truncate > 0 &&
          leftmost_questions_truncate < questions.size()) {
        KALDI_LOG << "Truncating " << questions.size() << " to "
                  << leftmost_questions_truncate << " for position 0.";
        phone_opts.initial_questions.assign(
            questions.begin(), questions.begin() + leftmost_questions_truncate);
      } else {
        phone_opts.initial_questions = questions;
      }
      qo.SetQuestionsOf(n, phone_opts);
    }

    QuestionsForKey pdfclass_opts(num_iters_refine);
    std::vector<std::vector<int32> > pdfclass_questions(max_num_pdfclasses-1);
    for (int32 i = 0; i < max_num_pdfclasses - 1; i++)
      for (int32 j = 0; j <= i; j++)
        pdfclass_questions[i].push_back(j);
    // E.g. if max_num_pdfclasses == 3,  pdfclass_questions is now [ [0], [0, 1] ].
    pdfclass_opts.initial_questions = pdfclass_questions;
    KALDI_LOG << "Setting questions for hmm-position [hmm-position ranges from 0 to "<< (max_num_pdfclasses-1) <<"]";
    qo.SetQuestionsOf(kPdfClass, pdfclass_opts);

    WriteKaldiObject(qo, questions_out_filename, binary);
    KALDI_LOG << "Wrote questions to "<<questions_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

