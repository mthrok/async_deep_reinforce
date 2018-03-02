from __future__ import print_function

import os
import errno
import struct
import tarfile
import datetime

from io import BytesIO


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
            'state': BytesIO(),
            'reward': BytesIO(),
            'action': BytesIO(),
        }

    def add(self, state, action=None, reward=None):
        self._objs['state'].write(state.tobytes())
        if action is not None:
            self._objs['action'].write(struct.pack('I', action))
        if reward is not None:
            self._objs['reward'].write(struct.pack('f', reward))

    def flush(self):
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.tgz'
        filepath = os.path.join(self.dir_path, filename)

        print('Saving record:', filepath)
        with tarfile.open(name=filepath, mode='w:gz') as tarobj:
            for key, value in self._objs.items():
                tarinfo = tarfile.TarInfo(key)
                tarinfo.size = value.tell()
                value.seek(0)
                tarobj.addfile(tarinfo, fileobj=value)
