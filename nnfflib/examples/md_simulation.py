import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from nnfflib.md import NVE, NVT, NPT
from nnfflib.schnet import SchNet
from nnfflib.help_functions import load_xyz
from yaff import *


def load_data(filename, verbose = False):
    for data in load_xyz(filename):
        positions = data['pos']
        numbers = data['Z']
        if 'Lattice' in data.keys():
            rvec = data['Lattice'].reshape([3, 3])
        else:
            rvec = None
            
        break
    
    return positions, numbers, rvec
  
    
if __name__ == '__main__':
    model = SchNet.from_restore_file('model_name', float_type = 64)
    
    filename = 'start_configuration.xyz'
    positions, numbers, rvec = load_data(filename)
    
    if rvec is None:
        system = System(numbers, positions * angstrom)
    else:
        system = System(numbers, positions * angstrom, rvecs = (rvec * angstrom).astype(np.float))
    
    NVT(system, model, 100000, screenprint = 1000, nprint = 1, dt = 0.5, temp = 300, name = 'md/test')
    
