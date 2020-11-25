"""
hyper parameter
"""

"""LSTM"""
BURN_IN_LENGTH = 40
SEQUENCE_LENGTH = 80
SEQUENCE_OVERLAP = 40

"""PER: Prioritized Experience Replay"""
N_STEP_BOOTSTRAPING = 5
PRIORITY_EXPONENT = 0.9 # Priority Exponent
IMPORTANCE_SAMPlING_EXPONENT = 0.6 # Importance Sampling Exponent