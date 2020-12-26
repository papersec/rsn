"""
rl_function
"""

import numpy as np
import torch

from rsn.util.rescale_q import inverse_rescale, rescale
from rsn.util import make_1darray

from rsn.hyper_parameter import DISCOUNT_FACTOR, N_STEP_BOOTSTRAPING

def replay_q(q, device, sequence, n_action, hidden=None, no_grad=False):
    """
    given ARHS sequence들을 q 신경망에 입력하여 출력 반환
    hidden 이 주어지지 않으면 sequence 첫 ARHS의 h 사용

    q: q 네트워크
    hidden - ndarray[BATCH_SIZE], dtype=(Tensor[HIDDEN_SIZE], Tensor[HIDDEN_SIZE])
    sequence - ndarray[SEQ_LEN, BATCH_SIZE], dtype=ARHS
    """

    seq_len, batch_size = sequence.shape

    sequence = sequence.reshape(seq_len * batch_size)

    a, r, h, s = (make_1darray(e) for e in zip(*sequence))

    a = a.astype(np.int64)
    a = torch.from_numpy(a)
    a_tensor = torch.zeros(size=(seq_len*batch_size, n_action), dtype=torch.int64)
    a_tensor[torch.arange(seq_len*batch_size), a] = 1
    a = a_tensor.reshape(seq_len, batch_size, n_action)
    # Now type of a is Tensor[SEQ_LEN, BS, N_ACTION]

    r = r.reshape(seq_len, batch_size).astype(np.float32)
    r = torch.from_numpy(r).unsqueeze(2).float().float().to(device)
    # Now type of r is Tensor[SEQ_LEN, BS, 1]

    if hidden is None:
        h = h[:batch_size]
        h = (torch.stack(e).float().to(device) for e in zip(*h))
    else:
        h = hidden
    
    # Now type of h is (Tensor[BS, HIDDEN_SIZE], Tensor[BS, HIDDEN_SIZE])

    s = torch.stack(tuple(s))
    s = s.reshape(seq_len, batch_size, *s.shape[1:]).float().to(device)
    # Now type of s is Tensor[SEQ_LEN, BS, FRAMESTACK, SCR_H, SCR_W]

    if no_grad:
        with torch.no_grad():
            q_values, hidden = q(s, h, a, r)
    else:
        q_values, hidden = q(s, h, a, r)
    
    return q_values, hidden


def compute_td_error(experiences, q, q_target, device, n_action, no_grad=False):
    """
    experiences들이 주어졌을 때 각각의 TD error 계산
    """
    burnin, sequence, bootstrap = zip(*[e.decompress() for e in experiences])

    burnin, sequence, bootstrap = (np.stack(e).transpose(1, 0) for e in (burnin, sequence, bootstrap))

    action = torch.tensor(tuple(b.a for b in bootstrap[0]), dtype=torch.int64)

    bootstrap_n = len(bootstrap)
    bootstrap_reward = tuple(zip(*bootstrap.reshape(-1)))[1]
    bootstrap_reward = torch.tensor(bootstrap_reward, dtype=torch.float32).reshape(bootstrap_n, -1) # Tensor[N_STEP_BOOTSTRAP, BATCH_SIZE]

    _, hidden = replay_q(q, device, burnin, n_action, no_grad=True)
    q_values, hidden = replay_q(q, device, sequence, n_action, hidden=hidden, no_grad=no_grad)

    batch_size = q_values.shape[0]
    q_expect = q_values[torch.arange(batch_size), action]
    
    q_values, _ = replay_q(q, device, bootstrap, n_action, hidden=hidden, no_grad=True)
    q_argmax = torch.argmax(q_values, dim=1)

    _, hidden = replay_q(q_target, device, burnin, n_action, no_grad=True)
    _, hidden = replay_q(q_target, device, sequence, n_action, hidden=hidden, no_grad=True)
    q_values, _ = replay_q(q_target, device, bootstrap, n_action, hidden=hidden, no_grad=True)

    q_target = q_values[torch.arange(batch_size), q_argmax]
    q_target = inverse_rescale(q_target)

    with torch.no_grad():
        for i in reversed(range(N_STEP_BOOTSTRAPING)):
            q_target = q_target * DISCOUNT_FACTOR
            q_target = q_target + bootstrap_reward[i,:]
    
    q_target = rescale(q_target)
    return q_expect - q_target
