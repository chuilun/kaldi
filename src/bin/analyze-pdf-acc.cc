// bin/analyze-pdf-acc.cc

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

/** @brief Sums the pdf vectors to counts, this is used to obtain prior counts for hybrid decoding.
*/
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

#include <iomanip>
#include <algorithm>
#include <numeric>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::uint64 uint64;
  try {
    const char *usage =
        "Computes element counts from integer vector table.\n"
        "(e.g. get each pdf-accuracy to estimate DNN-output posterior "
        "for data analysis)\n"
        "Usage: analyze-pdf-acc [options] <DNN-output-posterior-rspecifier> <targets-label-rspecifier>\n"
        "e.g.: analyze-pdf-acc ark:mlpout.ark ark:1.ali\n";

    ParseOptions po(usage);

    bool binary = false;
    std::string symbol_table_filename = "";

    po.Register("binary", &binary, "write in binary mode");
    po.Register("symbol-table", &symbol_table_filename,
                "Read symbol table for display of counts");

    int32 counts_dim = 0;
    po.Register("counts-dim", &counts_dim,
                "Output dimension of the counts, "
                "a hint for dimension auto-detection.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string feature_rspecifier = po.GetArg(1),
    		targets_rspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    // Buffer for accumulating the counts
    Vector<double> counts(counts_dim, kSetZero);

    // Buffer for accumulating the correct counts
    Vector<double> TP(counts_dim, kSetZero);
    Vector<double> FN(counts_dim, kSetZero);
    Vector<double> FP(counts_dim, kSetZero);
    //Vector<double> TN(counts_dim, kSetZero);
    Vector<double> precision(counts_dim, kSetZero);
    Vector<double> recall(counts_dim, kSetZero);


    int32 num_done = 0, num_other_error = 0;
    int32 rows, cols, id;
    BaseFloat max = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
    	// read
    	const Matrix<BaseFloat> &mat = feature_reader.Value();
    	std::string utt = feature_reader.Key();
    	rows = mat.NumRows();
    	cols = mat.NumCols();

    	if (cols > counts.Dim()) {
    		counts.Resize(cols, kCopyData);
    		TP.Resize(cols,kCopyData);
    		FN.Resize(cols,kCopyData);
    		FP.Resize(cols,kCopyData);
    		precision.Resize(cols,kCopyData);
    		recall.Resize(cols,kCopyData);
    		//TN.Resize(cols,kCopyData);
    	}
    	if (!targets_reader.HasKey(utt)) {
    	  KALDI_WARN << utt << ", missing targets";
    	  num_other_error++;
    	  continue;
    	}
    	const std::vector<int32> &target = targets_reader.Value(utt);
    	if (rows != target.size()) {
    		KALDI_WARN << utt << ", length mismatch of targets label" << target.size()
    						   << " and posterior matrix " << rows;
    		num_other_error++;
    		continue;
    	}

    	// Accumulate the counts
    	for (int i = 0; i < rows; i++) {
            max = 0; id = 0;
    		for (int j = 0; j < cols; j++) {
    			if (max < mat(i,j)) {
    				max = mat(i,j);
    				id = j;
    			}
    		}

    		KALDI_ASSERT(target[i] >= 0 && target[i] < cols);
    		if (target[i] == id)
    			TP(target[i]) += 1.0;
    		else
    			FP(id) += 1.0;
    		counts(target[i]) += 1.0;
    	}
    	num_done++;
    }

    for (size_t i = 0; i < precision.Dim(); i++) {
    	precision(i) = (TP(i)+FP(i)) != 0 ? TP(i)/(TP(i)+FP(i)) : 0;
    	recall(i) = counts(i) != 0 ? TP(i)/counts(i) : 0;
    }

    // Report elements with zero counts
    for (size_t i = 0; i < counts.Dim(); i++) {
      if (0.0 == counts(i)) {
        KALDI_WARN << "Zero count for label " << i << ", this is suspicious.";
      }
    }

    //
    // THE REST IS FOR ANALYSIS, IT GETS PRINTED TO LOG
    //
    if (symbol_table_filename != "" || (kaldi::g_kaldi_verbose_level >= 0)) {
      // load the symbol table
      fst::SymbolTable *elem_syms = NULL;
      if (symbol_table_filename != "") {
          elem_syms = fst::SymbolTable::ReadText(symbol_table_filename);
          if (!elem_syms)
            KALDI_ERR << "Could not read symbol table from file "
                      << symbol_table_filename;
      }

      // sort the counts
      std::vector<std::pair<double, int32> > sorted_counts;
      for (int32 i = 0; i < counts.Dim(); i++) {
        sorted_counts.push_back(
                        std::make_pair(static_cast<double>(counts(i)), i));
      }
      std::sort(sorted_counts.begin(), sorted_counts.end());

      std::ostringstream os;
      double sum = counts.Sum();

      os << "Printing...\n### The sorted count table," << std::endl;
      os << "count\t\t(norm),\tid\t(symbol),\tprecision(tp/(tp+fp))\trecall(tp/(tp+fn)):" << std::endl;
      for (int32 i = 0; i < sorted_counts.size(); i++) {
        os << sorted_counts[i].first << "\t\t("
           << static_cast<float>(sorted_counts[i].first) / sum << "),\t"
           << sorted_counts[i].second << "\t"
           << (elem_syms != NULL ? "(" +
                           elem_syms->Find(sorted_counts[i].second) + ")" : "") << "\t"
		   << precision(sorted_counts[i].second)*100 << "%\t"
		   << recall(sorted_counts[i].second)*100 << "%"
           << std::endl;
      }
      os << "\n#total " << sum
         << " (" << static_cast<float>(sum)/100/3600 << "h) "
         << TP.Sum()/sum*100 << "% recall "
         << std::endl;
      KALDI_LOG << os.str();
    }

    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts, "
              << "skipped " << num_other_error << " vectors.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
