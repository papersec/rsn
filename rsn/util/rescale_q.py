"""
rescale_q
"""

import torch

from rsn.hyper_parameter import Q_RESCALE_FACTOR

def rescale(x):
    return x.sign() * (torch.sqrt(x.abs() + 1.0) - 1.0) + Q_RESCALE_FACTOR * x

def inverse_rescale(x):
    return x.sign() * (Q_RESCALE_FACTOR * (x.abs() + 1.0) + 0.5 - torch.sqrt((Q_RESCALE_FACTOR * (x.abs() + 1.0) + 0.5)**2 \
    - (Q_RESCALE_FACTOR**2) * (x**2 + 2*(x.abs())))) / (Q_RESCALE_FACTOR**2)