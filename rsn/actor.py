"""
actor
"""

from collections import deque, namedtuple

import numpy as np
import torch

from rsn.network import DRQN
from rsn.hyper_parameter import ACTOR_EXPERIENCE_BUFFER_SIZE
from rsn.hyper_parameter import BURN_IN_LENGTH, SEQUENCE_LENGTH, SEQUENCE_OVERLAP
from rsn.hyper_parameter import N_STEP_BOOTSTRAPING
SEQUENCE_REMAIN = (BURN_IN_LENGTH + N_STEP_BOOTSTRAPING) % SEQUENCE_OVERLAP

HSAR = namedtuple("HSAR", ['h', 's', 'a', 'r'])
SAR = namedtuple("SAR", ['s', 'a', 'r'])
Experience = namedtuple("Experience", ["hidden", "burnin", "sequence", "bootstrap"])

def _tie_sar(trajectory):
    """
    trajectory - list(SAR)
    """
    s, a, r = zip(*trajectory)
    # 이제 s, a, r 각각은 len(trajectory) 길이의 튜플
    a = np.array(a)
    r = np.array(r)

    return SAR(s, a, r)


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

    def act(self, q_values) -> int:
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
        return int(action)

    def run_episode(self):
        env = self.env
        trajectory = deque(maxlen=BURN_IN_LENGTH+SEQUENCE_LENGTH+N_STEP_BOOTSTRAPING)

        raw_obs = self.env.reset()
        hidden = (torch.zeros(size=(1, 512)), torch.zeros(size=(1, 512)))

        for t in range(1, 108000+1):
            # observation 전처리
            obs = torch.from_numpy(np.array(raw_obs) / 256)
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

            # action 선택
            action = self.act(q_values.squeeze(0))

            # action 실행
            reward, next_obs, done, _ = env.step(action)

            # trajectory update
            hsar = HSAR(
                h = hidden,
                s = raw_obs,
                a = action,
                r = reward,
            )
            trajectory.append(hsar)

            # Add Experience to Local buffer
            if t % SEQUENCE_OVERLAP == SEQUENCE_REMAIN and t >= (BURN_IN_LENGTH+SEQUENCE_LENGTH+N_STEP_BOOTSTRAPING):
                burnin = trajectory[0:BURN_IN_LENGTH]
                sequence = trajectory[BURN_IN_LENGTH:BURN_IN_LENGTH+SEQUENCE_LENGTH]
                bootstrap = trajectory[BURN_IN_LENGTH+SEQUENCE_LENGTH:]

                burnin, sequence, bootstrap = list(map(_tie_sar, (burnin, sequence, bootstrap)))

                e = Experience(
                    hidden=trajectory[0].hidden,
                    burnin=burnin,
                    sequence=sequence,
                    bootstrap=bootstrap,
                )

                self.local_buffer.append(e)

            # Send Experiences
            if len(self.local_buffer >= ACTOR_EXPERIENCE_BUFFER_SIZE):
                # Calculate TD ERROR
                # Burn in
                # Main Sequence
                # expected Q

                pass

            # obs <- next_obs
            # Parameter Update

            # if done, break


