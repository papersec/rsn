"""
parameter server

Actor: Get parameter
Learner: Set parameter
"""

import ray

@ray.remote
class _RemoteParameterServer:
    
    def __init__(self):
        self.q = None
        self.q_target = None
    
    def upload(self, q, q_target):
        self.q = q
        self.q_target = q_target

    def download(self):
        return self.q, self.q_target


class ParameterServer:

    def __init__(self):
        self.ps = _RemoteParameterServer.remote()
    
    def upload(self, q, q_target):
        ref = self.ps.upload.remote(q, q_target)
        return ray.get(ref)

    def download(self):
        return ray.get(self.ps.download.remote())