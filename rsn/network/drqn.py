import torch.nn as nn
import torch.nn.functional as F

class DRQN(nn.Module):
    """
    Deep Recurrent Q Network

    Actor -  한 framestack씩 입력되고, 이전 trajectory까지의 true hidden state가 제공됨
    Learner - 80 framestack과 stored state가 제공됨
    """

    def __init__(self, n_action):
        super(DRQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(7*7*64 + n_action + 1, 512)

        self.fc_v_0 = nn.Linear(512, 512)
        self.fc_v_1 = nn.Linear(512, 1)
        self.fc_a_0 = nn.Linear(512, 512)
        self.fc_a_1 = nn.Linear(512, n_action)
    
    def forward(self, fs, hidden, action, reward):
        """
        fs (=framestack) - Tensor[SEQ_LEN, BS, STACK, SCR_W, SCR_H]
        hidden - tuple(Tensor[BS, HIDDEN_SIZE], Tensor[BS, HIDDEN_SIZE])
            hidden state와 cell state의 tuple
        action - Tensor[SEQ_LEN, BS, N_ACTION]
            previous timestep의 one hot encoded action
        reward - Tensor[SEQ_lEN, BS, 1]
            previous timestep의 one hot encoded reward (without clipping)
        * All inputs must be on the same device.
        """
        seq_len, bs = fs.shape[0:2]

        x = fs.view(seq_len*bs, *fs.shape[2:])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conf2(x))
        x = F.relu(self.conv3(x))
        # now shape of x is [SEQ_LEN*BS, 7, 7, 64]

        x = x.view(seq_len, bs, -1)
        x = torch.cat((x, action, reward), dim=2)

        # LSTM hidden state input has dim0=num_layers*num_directions which is 1
        h_0, c_0 = (h.unsqueeze(0) for h in hidden)

        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # shape of x is [SEQ_LEN, BS, HIDDEN_SIZE] (ignore)
        # shape of h_n, c_n is [1, BS, HIDDEN_SIZE]
        h_n, c_n = [h.squeeze(0) for h in (h_n, c_n)]

        v = self.fc_v_1(F.relu(self.fc_v_0(h_n)))
        a = self.fc_a_1(F.relu(self.fc_a_0(h_n)))
        return self.duel(v, a), (h_n, c_n)

    def duel(self, v, a):
        """
        a - Tensor[BS, N_ACTION]
        v - Tensor[BS, 1]
        """
        q = v + a - a.mean(dim=1, keepdim=True)
        return q.squeeze(1)