// util/circular-queue.h

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

#ifndef KALDI_UTIL_CIRCULAR_QUEUE_H_
#define KALDI_UTIL_CIRCULAR_QUEUE_H_

#include <vector>
#include <list>
#include "util/stl-utils.h"

namespace kaldi {

template<class T>
class CircularQueue {
public:
	CircularQueue(int size = 1);

	void Push();

	void Pop();

	T* Front();

	T* Back();

	int Size();

	bool Empty();

	int Capacity();

	void Clear();

	void Resize(int size = 1);
private:

	std::list<T> buffer_;
	typename std::list<T>::iterator front_;
	typename std::list<T>::iterator rear_;
	typename std::list<T>::iterator back_;
	int	 size_;
};

}

#include "util/circular-queue-inl.h"

#endif /* UTIL_CIRCULAR_QUEUE_H_ */
