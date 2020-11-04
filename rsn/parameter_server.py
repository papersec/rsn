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
    
    def update(self, q, q_target):
        self.q = q
        self.q_target = q_target

    def get(self):
        return self.q, self.q_target


class ParameterServer:

    def __init__(self):
        self.ps = _RemoteParameterServer.remote()
    
    def upload(self, q, q_target, wait=True):
        ref = self.ps.upload.remote(q, q_target)
        if wait:
            return ray.get(ref)
        else:
            return ref

    def download(self):
        return ray.get(self.ps.download.remote())