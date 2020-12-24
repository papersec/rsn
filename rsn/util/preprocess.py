import numpy as np
import torch

def preprocess_observation(raw_obs):
    """
    raw_obs: Lazyframes
    """
    obs = torch.from_numpy(np.array(raw_obs) / 256)
    obs = obs.view(1, 1, *obs.shape)
    return obs

def reward_to_tensor(r):
    """
    r - list[BS, SEQ_LENGTH]
    """
    if type(r) is float:
        r = [[r]]
    assert type(r) is list
    if type(r[0]) is float:
        r = [r]
    assert type(r[0]) is list

    return torch.tensor(r).permute(1,0).unsqueeze(2)

def action_to_tensor(a, n_action):
    """
    a - list[BS, SEQ_LENGTH]
    """

    if type(a) is int:
        a = [[a]]
    assert type(a) is list
    if type(a[0]) is int:
        a = [a]
    assert type(a[0]) is list

    a = torch.tensor(a).permute(1,0)
    seq_length, batch_size = a.shape
    a = a.view(seq_length,*batch_size)

    a_tensor = torch.zeros(size=(seq_length*batch_size, n_action), dtype=torch.int64)
    a_tensor[torch.arange(seq_length*batch_size), a] = 1

    return a_tensor.view(seq_length, batch_size, n_action)