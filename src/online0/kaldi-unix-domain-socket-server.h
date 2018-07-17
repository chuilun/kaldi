// online0/kaldi-unix-domain-socket-server.h

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

#ifndef ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_SERVER_H_
#define ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_SERVER_H_

#include <sys/un.h>
#include <sys/socket.h>	/* basic socket definitions */
#include <fcntl.h>		/* for nonblocking */
#include <unistd.h>

#include "base/kaldi-error.h"
#include "online0/kaldi-unix-domain-socket.h"

class UnixDomainSocketServer {
public:
	UnixDomainSocketServer(std::string unix_filepath, int type = SOCK_STREAM)
	{
		if ((socket_ = socket(AF_LOCAL, type, 0)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
				KALDI_ERR << "Error open socket , errno was: " << c;
		}

		// socket address
		unlink(unix_filepath.c_str());
		bzero(&socket_addr_, sizeof(sockaddr_un));
		socket_addr_.sun_family = AF_LOCAL;
		strcpy(socket_addr_.sun_path, unix_filepath.c_str());

		// bind socket with a absolute local file path
		if (bind(socket_, (struct sockaddr*)&socket_addr_, sizeof(sockaddr_un)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error bind socket with path " << unix_filepath << " , errno was: " << c;
		}

		// listen socket
		if (listen(socket_, 1024) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error listen socket with path " << unix_filepath << " , errno was: " << c;
		}
	}

	UnixDomainSocketServer():socket_(-1) {}

	UnixDomainSocketServer(int type = SOCK_STREAM)
	{
		if ((socket_ = socket(AF_LOCAL, type, 0)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
				KALDI_ERR << "Error open socket , errno was: " << c;
		}
	}

	void Bind(std::string unix_filepath)
	{
		// socket address
		unlink(unix_filepath.c_str());
		bzero(&socket_addr_, sizeof(sockaddr_un));
		socket_addr_.sun_family = AF_LOCAL;
		strcpy(socket_addr_.sun_path, unix_filepath.c_str());

		// bind socket with a absolute local file path
		if (bind(socket_, (struct sockaddr*)&socket_addr_, sizeof(sockaddr_un)) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error bind socket with path " << unix_filepath << " , errno was: " << c;
		}

		// listen socket
		if (listen(socket_, 1024) < 0) {
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error listen socket with path " << unix_filepath << " , errno was: " << c;
		}
	}

	~UnixDomainSocketServer()
	{
		close(socket_);
	}

	UnixDomainSocket* Accept(bool block = true)
	{
		int conn_socket_fd;
		struct sockaddr_un client_socket_addr;
		socklen_t addr_len;

		/*
		if ((conn_socket_fd = accept(socket_, (struct sockaddr*)&client_socket_addr, &addr_len)) < 0)
		{
			const char *c = strerror(errno);
			if (c == NULL) { c = "[NULL]"; }
			KALDI_ERR << "Error accept socket, errno was: " << c;
		}*/

		if ((conn_socket_fd = accept(socket_, (struct sockaddr*)&client_socket_addr, &addr_len)) < 0)
			return NULL;

		// non block connect
		if (!block) {
			int flags = fcntl(conn_socket_fd, F_GETFL, 0);
			fcntl(conn_socket_fd, F_SETFL, flags | O_NONBLOCK);
		}

        int buffer_size = 4194304;
        setsockopt(conn_socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(int));
        setsockopt(conn_socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(int));

		UnixDomainSocket *socket = new UnixDomainSocket(conn_socket_fd, socket_addr_, block);
		return socket;
	}

	void Close()
	{
		close(socket_);
	}

	bool isClosed()
	{
		int error = 0;
		socklen_t len = sizeof(error);
		int ret = getsockopt(socket_, SOL_SOCKET, SO_ERROR, &error, &len);
		if (error != 0 || ret != 0)
			return true;
		return false;
	}

private:
	int socket_; // listening socket
	struct sockaddr_un	socket_addr_;

};




#endif /* ONLINE0_KALDI_UNIX_DOMAIN_SOCKET_SERVER_H_ */
