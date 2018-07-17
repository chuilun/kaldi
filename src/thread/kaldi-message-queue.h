// thread/kaldi-message-queue.h

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

#ifndef THREAD_KALDI_MESSAGE_QUEUE_H_
#define THREAD_KALDI_MESSAGE_QUEUE_H_ 1


#include <sys/stat.h>
#include <fcntl.h>
#include <mqueue.h>		/* Posix message queues */

#define	FILE_MODE	(S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)

namespace kaldi {

class MessageQueue {

public:
	MessageQueue(std::string pathname, int oflag, mode_t mode = FILE_MODE, struct mq_attr *attr = NULL);

	MessageQueue();

	~MessageQueue();

	void Open(std::string pathname, int oflag = O_RDWR);

	void Create(std::string pathname, struct mq_attr *attr, int oflag = O_RDWR | O_CREAT | O_EXCL);

	int Send(char *ptr, size_t len, unsigned int prio);

	ssize_t Receive(char *ptr, size_t len, unsigned int *prio);

	void Getattr(struct mq_attr *mqstat);

	void Setattr(struct mq_attr *mqstat, struct mq_attr *omqstat);

private:

	std::string pathname_;
	mqd_t	mqd_;
	int oflag_;
	mode_t mode_;
	mq_attr *attr_;
    KALDI_DISALLOW_COPY_AND_ASSIGN(MessageQueue);
};


}

#endif /* THREAD_KALDI_MESSAGE_QUEUE_H_ */
