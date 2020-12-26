"""
actor
"""

from collections import namedtuple

import numpy as np
import torch

from rsn.network import DRQN
from rsn.hyper_parameter import ACTOR_EXPERIENCE_BUFFER_SIZE
from rsn.hyper_parameter import BURN_IN_LENGTH, SEQUENCE_LENGTH, SEQUENCE_OVERLAP, HIDDEN_STATE_DIMENSION
from rsn.hyper_parameter import N_STEP_BOOTSTRAPING
from rsn.util.experience import Experience
from rsn.util import ARHS
from rsn.util.make_1darray import make_1darray
from rsn.util.sliceable_deque import sliceable_deque
from rsn.network.rl_function import compute_td_error, replay_q

SEQUENCE_REMAIN = (BURN_IN_LENGTH + N_STEP_BOOTSTRAPING) % SEQUENCE_OVERLAP

class Actor:
    
    def __init__(self, env, n_action, eps, replay_memory, parameter_server):
        self.env = env
        self.n_action = n_action
        self.eps = eps
        self.replay_memory = replay_memory
        self.parameter_server = parameter_server

        self.local_buffer = []

        self.q = DRQN(n_action).float()
        self.q_target = DRQN(n_action).float()

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
        trajectory = sliceable_deque(maxlen=BURN_IN_LENGTH+SEQUENCE_LENGTH+N_STEP_BOOTSTRAPING)

        score = 0.0

        obs = torch.from_numpy(np.array(self.env.reset(), dtype=np.float32)/256).float()
        assert list(obs.shape) == [4, 84, 84]
        hidden = (torch.zeros(size=(HIDDEN_STATE_DIMENSION,)), torch.zeros(size=(HIDDEN_STATE_DIMENSION,)))
        prev_action = 0
        prev_reward = 0.0

        param_update_cnt = 0

        for t in range(1, 108000+1):

            param_update_cnt += 1
            
            # trajectory 업데이트
            arhs = ARHS(
                a=prev_action,
                r=prev_reward,
                h=hidden,
                s=obs,
            )
            trajectory.append(arhs)

            arhs_arr = np.empty(shape=(1,1), dtype=object)
            arhs_arr[:,:] = [[arhs]]

            q_values, hidden = replay_q(self.q, torch.device("cpu"), arhs_arr, self.n_action, no_grad=True)

            hidden = tuple(hc.squeeze(0) for hc in hidden) # Batch Size 1이므로 squeeze

            # action 선택
            action = self.act(q_values.squeeze(0))

            # action 실행
            next_raw_obs, reward, done, _ = env.step(action)
            score += reward

            # Add Experience to Local buffer
            if t % SEQUENCE_OVERLAP == SEQUENCE_REMAIN and t >= (BURN_IN_LENGTH+SEQUENCE_LENGTH+N_STEP_BOOTSTRAPING):
                burnin = make_1darray(trajectory[:BURN_IN_LENGTH])
                sequence = make_1darray(trajectory[BURN_IN_LENGTH:BURN_IN_LENGTH+SEQUENCE_LENGTH])
                bootstrap = make_1darray(trajectory[BURN_IN_LENGTH+SEQUENCE_LENGTH:])

                e = Experience(burnin, sequence, bootstrap)
                self.local_buffer.append(e)

            # Send Experiences
            if len(self.local_buffer) >= ACTOR_EXPERIENCE_BUFFER_SIZE:
                # Calculate TD ERROR
                device = torch.device("cpu")

                td_errors = compute_td_error(self.local_buffer, self.q, self.q_target, device, self.n_action, no_grad=True).data.tolist()
                self.replay_memory.add(self.local_buffer, td_errors)
                self.local_buffer = []

            # Edit variables for next timestep
            obs = torch.from_numpy(np.array(next_raw_obs, dtype=np.float32)/256).float()
            prev_action = action
            prev_reward = reward

            # Parameter Update
            if param_update_cnt % 400 == 0:
                param_update_cnt = 0

                param_q, param_q_target = self.parameter_server.download()
                if param_q is not None:
                    self.q.load_state_dict(param_q)
                if param_q_target is not None:
                    self.q_target.load_state_dict(param_q_target)

            # if done, break
            if done:
                trajectory.clear()
                break
        
        return score