"""
actor
"""

from collections import deque, namedtuple

import numpy as np
import torch

from rsn.network import DRQN
from rsn.hyper_parameter import SEQUENCE_LENGTH

Sar = namedtuple("Sar", ['s', 'a', 'r'])

class Actor:
    
    def __init__(self, env, n_action, eps, replay_memory, parameter_server):
        self.env = env
        self.n_action = n_action
        self.eps = eps
        self.replay_memory = replay_memory
        self.parameter_server = parameter_server

        self.local_buffer = []

        self.q = DRQN(n_action)
        self.q_target = DRQN(n_action)

    def act(self, q_values):
        """
        q_values - Tensor[N_ACTION]
        """

        # Epsilon Greedy
        rand = np.random.rand()

        if rand < self.eps:
            action = np.random.randint(self.n_action)
        else:
            # Q를 이용한 action 결정
            action = torch.argmax(q_values)
        return action

    def run_episode(self):
        trajectory = deque(maxlen=SEQUENCE_LENGTH)

        obs = self.env.reset()
        hidden = (torch.zeros(size=(1, 512)), torch.zeros(size=(1, 512)))

        for t in range(108000):
            # observation 전처리
            obs = torch.from_numpy(np.array(obs) / 256)
            obs = obs.view(1, 1, *obs.shape)

            # prev action, reward 가져오기
            prev_action = torch.zeros(size=(1, 1, self.n_action))
            prev_reward = torch.zeros(size=(1, 1, 1))
            if len(trajectory) == 0:
                prev_action[0,0,0] = 1.0 # NOOP
            else:
                prev_sar = trajectory[-1]
                prev_action[0,0,prev_sar.a] = 1.0
                prev_reward[0,0,0] = prev_sar.r

            self.q.eval()
            with torch.no_grad():
                q_values, hidden = self.q(obs, hidden, prev_action, prev_reward)

            action = self.act(q_values.squeeze(0))

            reward, next_obs, done, _ = env.step(action)

            # Add Experience to Local buffer
            # Send Experiences
            # obs <- next_obs
            # Parameter Update

            # if done, break


