# -*- coding: utf-8 -*-
import os
from ale_python_interface import ALEInterface

def _get_action_size(rom):
    print(rom)
    ale = ALEInterface()
    ale.loadROM(rom)
    n_actions = len(ale.getMinimalActionSet())
    del ale
    return n_actions

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp

INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # parallel thread size
ROM = os.environ.get('ROM', "roms/breakout.bin")
ACTION_SIZE = _get_action_size(ROM.encode('ascii'))
LOG_FILE = 'a3c/%s' % ROM.split('.')[0].split('/')[1]
CHECKPOINT_DIR = LOG_FILE

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 30 * 10**6
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
USE_LSTM = True # True for A3C LSTM, False for A3C FF
