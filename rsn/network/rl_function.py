"""
rl_function
"""


import torch

from rsn.util.preprocess import preprocess_observation, reward_to_tensor, action_to_tensor
from rsn.util.rescale_q import inverse_rescale, rescale
from rsn.hyper_parameter import DISCOUNT_FACTOR, N_STEP_BOOTSTRAPING

def replay_q(q, device, sequence, n_action, hidden=None, no_grad=False):
    """
    given ARHS sequence를 q 신경망에 입력하여 출력 반환
    hiddend 이 주어지지 않으면 첫 ARHS의 h 사용

    q: q 네트워크
    hidden - list((Tensor[HIDDEN_STATE_DIMENSION], Tensor[HIDDEN_STATE_DIMENSION]))
    sequence - list(ARHS)
    boolean
    """
    a, r, h, s = zip(*sequence)

    s = torch.stack([torch.stack([preprocess_observation(raw_obs) for raw_obs in s_]) for s_ in s])
    s = s.permute(1,0,2,3,4).to(device)
    # now type of s is Tensor[SEQ_LENGTH, BS, STACK, SCR_H, SCR_W]

    if hidden is None:
        h = zip(*[h_[0] for h_ in h])
    else:
        h = zip(*hidden)
    
    h = tuple(map(torch.stack, h))
    # now type of h is tuple(Tensor[BS, HIDDEN_SIZE], Tensor[BS, HIDDEN_SIZE])

    a_tensor = action_to_tensor(a, n_action)
    r_tensor = reward_to_tensor(r)

    if no_grad:
        with torch.no_grad():
            q_values, hidden = q(s, h, a_tensor, r_tensor)
    else:
        q_values, hidden = q(s, h, a_tensor, r_tensor)
    
    return q_values, hidden


def compute_td_error(experiences, q, q_target, device, n_action, no_grad=False):
    """
    experiences들이 주어졌을 때 각각의 TD error 계산
    """

    burnin = [e.burnin for e in experiences]
    sequence = [e.sequence for e in experiences]
    bootstrap = [e.bootstrap for e in experiences]

    action = torch.tensor([b[0].a for b in bootstrap], dtype=torch.int64)
    bootstrap_reward = torch.tensor([[arhs.r for arhs in b] for b in bootstrap]) # Tensor[BATCH_SIZE, N_STEP_BOOTSTRAP]

    _, hidden = replay_q(q, device, burnin, n_action, no_grad=True)
    q_values, hidden = replay_q(q, device, sequence, n_action, hidden=hidden, no_grad=no_grad)

    batch_size = q_values.shape[0]
    q_expect = q_values[torch.arange(batch_size), action]
    
    q_values, _ = replay_q(q, device, bootstrap, n_action, hidden=hidden, no_grad=True)
    q_argmax = torch.argmax(q_values, dim=1)

    _, hidden = replay_q(q_target, device, burnin, n_action, no_grad=True)
    _, hidden = replay_q(q_target, device, sequence, n_action, no_grad=True)
    q_values, _ = replay_q(q_target, device, bootstrap, n_action, no_grad=True)

    q_target = q_values[torch.arange(batch_size), q_argmax]
    q_target = inverse_rescale(q_target)

    with torch.no_grad():
        for i in reversed(range(N_STEP_BOOTSTRAPING)):
            q_target = q_target * DISCOUNT_FACTOR
            q_target = q_target + bootstrap_reward[:,i]
    
    q_target = rescale(q_target)
    return (q_expect - q_target)**2 * 0.5
