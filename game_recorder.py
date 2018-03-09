from __future__ import print_function

import os
import errno
import datetime

import numpy as np


class Recorder(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._objs = None

        print('Initializing recording directory:', dir_path)
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def init(self):
        self._objs = {
            'state': [],
            'reward': [],
            'action': [],
        }

    def add(self, state, action=None, reward=None):
        self._objs['state'].append(np.copy(state))
        if action is not None:
            self._objs['action'].append(action)
        if reward is not None:
            self._objs['reward'].append(reward)

    def flush(self):
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.npz'
        filepath = os.path.join(self.dir_path, filename)
        with open(filepath, 'wb') as fileobj:
            np.savez_compressed(
                fileobj,
                state=np.asarray(self._objs['state']),
                action=np.asarray(self._objs['action']),
                reward=np.asarray(self._objs['reward']),
            )
