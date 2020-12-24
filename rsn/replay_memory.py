"""
replay memory

Actor: Add experience
Learner: Get experience, Update priority
"""

import numpy as np
import ray

from rsn.hyper_parameter import PRIORITY_EXPONENT, IMPORTANCE_SAMPlING_EXPONENT

def _get_priority(td_error):
    # Priority = (abs(TD_ERROR))^ALPHA
    return np.power(np.absolute(td_error), PRIORITY_EXPONENT)

@ray.remote
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.cursor = 0
    
    def _propagate(self, idx):
        parent = (idx - 1) // 2

        child1 = parent * 2 + 1
        child2 = parent * 2 + 2

        self.tree[parent] = self.tree[child1] + self.tree[child2]

        if parent != 0:
            self._propagate(parent)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.cursor + self.capacity - 1

        self.data[self.cursor] = data
        self.update(idx, p)

        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, p):
        self.tree[idx] = p
        self._propagate(idx)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]
    
    def size(self):
        return self.size

@ray.remote
class _RemoteReplayMemory:

    def __init__(self, capacity):
        self.tree = SumTree.remote(capacity=capacity)
    
    def add(self, experience, td_error): # Actor -> ReplayMemory
        p = _get_priority(td_error)
        ray.get(self.tree.add.remote(p, experience))
    
    def sample(self, size): # ReplayMemory -> Learner
        tree_total = ray.get(self.tree.total.remote())

        batch = []
        indices = []
        priorities = []
        segment_length = tree_total / size

        for i in range(size):
            left = segment_length * i
            right = left + segment_length

            s = np.random.uniform(left, right)
            
            idx, priority, experience = ray.get(self.tree.get.remote(s))
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / tree_total
        weight = np.power(sampling_probabilities * ray.get(self.tree.size.remote()), -IMPORTANCE_SAMPlING_EXPONENT)
        weight /= weight.max()

        return batch, indices, weight

    def update_priority(self, idx, td_error): # Learner -> ReplayMemory
        p = _get_priority(td_error)
        ray.get(self.tree.update.remote(idx, p))

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.update_priority(idx, td_error)
    
    def size(self):
        return ray.get(self.tree.size.remote())


class ReplayMemory:

    def __init__(self, capacity):
        self.rm = _RemoteReplayMemory.remote(capacity)
    
    def add(self, experience, td_error):
        if type(experience) is list: # 여러 experience add
            for e, t in zip(experience, td_error):
                self.add(e, t)
        return ray.get(self.rm.add.remote(experience, td_error))

    def sample(self, size):
        return ray.get(self.rm.sample.remote(size))
    
    def update_priority(self, idx, td_error):
        return ray.get(self.rm.update_priority.remote(idx, td_error))
    
    def update_priorities(self, indices, td_errors):
        return ray.get(self.rm.update_priorities.remote(indices, td_errors))

    def size(self):
        return ray.get(self.rm.size.remote())