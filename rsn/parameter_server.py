"""
parameter server

Actor: Get parameter
Learner: Set parameter
"""

import socket

import ray

@ray.remote
class ParameterServerBase:
    
    def __init__(self):
        self.q = None
        #self.q_target = None
    
    def upload(self, x):
        self.q = x

    def download(self):
        return self.q
    

@ray.remote(num_cpus=1)
class ParameterServerSocket:

    def __init__(self, psb):
        self.psb = psb # ParameterServerBase

        host, port = "localhost", 7020
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen()
        self.client_socket, self.address = self.server_socket.accept()
        print("Connected with", self.address)
    
    def step(self):
        msg = self.client_socket.recv(1024)
        print("Received", msg.decode(), "from", self.address)
        ret_msg = ray.get(self.psb.download.remote()).encode()
        self.client_socket.sendall(ret_msg)
    
    def start(self):
        while True:
            self.step()
    
    def close(self):
        self.client_socket.close()
        self.server_socket.close()


class ParameterServer:

    def __init__(self):
        self.psb = ParameterServerBase.remote()
        self.pss = ParameterServerSocket.remote(self.psb)
        self.pss_ref = self.pss.start.remote()
    
    def upload(self, x):
        ray.get(self.psb.upload.remote(x))

    def close(self):
        ray.get(self.pss.close.remote())


@ray.remote(num_cpus=1)        
class ParameterClientSocket:

    def __init__(self):
        host, port = "10.128.0.2", 7020
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))

    def download(self):
        client_socket.sendall("sendme".encode())
        data = client_socket.recv(1024)
        print("Received", data.decode())
    
    def close(self):
        self.client_socket.close()


class ParameterClient:

    def __init__(self):
        self.pcs = ParameterClientSocket()

    def download(self):
        ray.get(self.pcs.download.remote())
    
    def close(self):
        ray.get(self.pcs.close.remote())
