"""
learner
"""

import torch
import torch.optim as optim

from rsn.network import DRQN
from rsn.network.rl_function import compute_td_error
from rsn.hyper_parameter import BATCH_SIZE, TARGET_NEETWORK_COPY_PERIOD

class Learner:

    def __init__(self, n_action, replay_memory, parameter_server):
        self.n_action = n_action
        self.replay_memory = replay_memory
        self.parameter_server = parameter_server

        self.q = DRQN(n_action).float()
        self.q_target = DRQN(n_action).float()

        #assert torch.cuda.is_available()
        self.device = torch.device("cpu")

        self.optimizer = optim.Adam(self.q.parameters(), lr=1.0e-4, eps=1.0e-3)

    def learn(self):
        experiences, indices, weights = self.replay_memory.sample(BATCH_SIZE)

        # Need to be refactored, Given array is not writable
        weights = torch.from_numpy(weights.copy()).to(self.device)

        # Compute Loss
        td_errors = compute_td_error(experiences, self.q, self.q_target, self.device, self.n_action, no_grad=False)
        losses = (td_errors ** 2) * 0.5

        loss = (losses * weights).mean()

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_memory.update_priorities(indices, td_errors.data.tolist())

    def run(self):
        t = 0
        while True:
            t += 1
            if t % 10 == 0:
                print("Learning Epoch", t)

            self.learn()
            
            # Periodically update target network parameters from online network
            if t % TARGET_NEETWORK_COPY_PERIOD == 0:
                self.q_target.load_state_dict(self.q.state_dict())

            # Update parameter server
            param_q = self.q.state_dict()
            param_q_target = self.q_target.state_dict()

            self.parameter_server.upload(param_q, param_q_target)



