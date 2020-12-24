"""
actor
"""

from collections import deque, namedtuple

import numpy as np
import torch

from rsn.network import DRQN
from rsn.hyper_parameter import ACTOR_EXPERIENCE_BUFFER_SIZE
from rsn.hyper_parameter import BURN_IN_LENGTH, SEQUENCE_LENGTH, SEQUENCE_OVERLAP, HIDDEN_STATE_DIMENSION
from rsn.hyper_parameter import N_STEP_BOOTSTRAPING
from rsn.util.preprocess import preprocess_observation, action_to_tensor, reward_to_tensor
from rsn.network.rl_function import compute_td_error, replay_q

SEQUENCE_REMAIN = (BURN_IN_LENGTH + N_STEP_BOOTSTRAPING) % SEQUENCE_OVERLAP

ARHS = namedtuple("ARHS", ['a', 'r', 'h', 's'])
Experience = namedtuple("Experience", ["burnin", "sequence", "bootstrap"])

def _tie_arhs(trajectory):
    """
    trajectory - list(HSAR)
    """
    a, r, h, s = zip(*trajectory)
    # 이제 a, r, h, s 각각은 len(trajectory) 길이의 튜플
    a = np.array(a)
    r = np.array(r)

    return ARHS(a, r, h, s)


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
        print("Init Complete")

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
        hidden = (torch.zeros(size=(1, HIDDEN_STATE_DIMENSION)), torch.zeros(size=(1, HIDDEN_STATE_DIMENSION)))
        prev_action = 0
        prev_reward = 0.0

        for t in range(1, 108000+1):
            print("t", t)
            # observation 전처리
            obs = preprocess_observation(raw_obs)
            
            # trajectory 업데이트
            arhs = ARHS(
                a=prev_action,
                r=prev_reward,
                h=hidden,
                s=raw_obs
            )
            trajectory.append(arhs)

            q_values, hidden = replay_q(self.q, torch.device("cpu"), arhs, self.n_action, no_grad=True)

            # action 선택
            action = self.act(q_values.squeeze(0))

            # action 실행
            reward, next_raw_obs, done, _ = env.step(action)

            # Add Experience to Local buffer
            if t % SEQUENCE_OVERLAP == SEQUENCE_REMAIN and t >= (BURN_IN_LENGTH+SEQUENCE_LENGTH+N_STEP_BOOTSTRAPING):
                burnin = trajectory[0:BURN_IN_LENGTH]
                sequence = trajectory[BURN_IN_LENGTH:BURN_IN_LENGTH+SEQUENCE_LENGTH]
                bootstrap = trajectory[BURN_IN_LENGTH+SEQUENCE_LENGTH:]

                burnin, sequence, bootstrap = list(map(_tie_arhs, (burnin, sequence, bootstrap)))

                e = Experience(
                    burnin=burnin,
                    sequence=sequence,
                    bootstrap=bootstrap,
                )

                self.local_buffer.append(e)

            # Send Experiences
            if len(self.local_buffer) >= ACTOR_EXPERIENCE_BUFFER_SIZE:
                # Calculate TD ERROR
                device = torch.device("cpu")

                td_errors = compute_td_error(self.local_buffer, self.q, self.q_target, device, self.n_action, no_grad=True).data
                self.replay_memory.add(self.local_buffer, td_errors)

            raw_obs = next_raw_obs
            prev_action = action
            prev_reward = reward

            # Parameter Update
            if t % 400 == 0:
                param_q, param_q_target = self.parameter_server.download()
                if param_q is not None:
                    self.q.load_state_dict(param_q)
                if param_q_target is not None:
                    self.q_target.load_state_dict(param_q_target)

            # if done, break
            if done:
                break
