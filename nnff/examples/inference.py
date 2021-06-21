import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from nnff.help_functions import load_xyz, XYZLogger
from glob import glob
from nnff.schnet import SchNet
import time


def RMSE(residuals):
    return np.sqrt(np.sum(residuals**2) / len(residuals))
    

def analyse(energy, forces, target_energies, target_forces, num_numbers):
    energy = np.array(energy).reshape([-1])
    target_energies = np.array(target_energies).reshape([-1])
    
    num_numbers = np.array(num_numbers).reshape([-1])
    
    forces = np.concatenate(forces, axis = 0).reshape([-1])
    target_forces = np.concatenate(target_forces, axis = 0).reshape([-1])
    
    print('')
    print('----- Total errors ------')
    mae_energy = np.mean(np.abs(energy - target_energies))
    print('MAE energy: %f eV' % mae_energy)
    std_energy = RMSE(energy - target_energies)
    print('RMSE energy: %f eV' % std_energy)
    print('')
    
    print('----- Local Errors ------')
    mae_energy = np.mean(np.abs((energy - target_energies) / num_numbers))
    print('MAE atomic energy: %f eV' % mae_energy)
    std_energy = RMSE((energy - target_energies) / num_numbers)
    print('RMSE atomic energy: %f eV' % std_energy)
    mae_forces = np.mean(np.abs(forces - target_forces))
    print('MAE forces: %f eV / A' % mae_forces)
    std_forces = RMSE(forces - target_forces)
    print('RMSE forces: %f eV / A' % std_forces)
    print('')
    
    mae_targets = np.mean(np.abs(target_forces - np.mean(target_forces)))
    rmse_targets = RMSE(target_forces - np.mean(target_forces))
    print('Intrinsic force RMSE: %f eV / A' % rmse_targets)
    print('Intrinsic force MAE: %f eV / A' % mae_targets)
    print('')
            
    return mae_forces, mae_targets
    
    
if __name__ == '__main__':
    # Percentage of configs to validate
    analyse_rate = 0.25

    model = SchNet.from_restore_file('model_name', float_type = 32, reference = 0.)

    filenames = glob('validation_files/*.xyz')
    filenames.sort()

    energies = [] # Energy deviations
    forces = [] # Force deviations

    target_forces = []
    target_energies = []

    num_numbers = []

    start = time.time()
    index = 0
    analysed = 0
    for filename in filenames:
        print(filename)
        for data in load_xyz(filename):
            energy = data['energy']
            rvec = data['Lattice'].reshape([3, 3])
            numbers = data['Z']
            positions = data['pos']
            forces = data['force']
            
            index += 1
            
            if index % 2500 == 0:
                print(index)
            
            if 1. * analysed / index < analyse_rate:
                analysed += 1
            else:
                continue

            output = model.compute(positions, numbers, rvec = rvec, list_of_properties = ['energy', 'forces']) 
            schnet_energy = output['energy']
            schnet_forces = output['forces']
                
            energies.append(schnet_energy)
            forces.append(schnet_forces.flatten())
            
            target_forces.append(forces.flatten())
            target_energies.append(energy)
            
            num_numbers.append(len(numbers))

    analyse(energies, forces, target_energies, target_forces, num_numbers)
            
    print('Total number of configs encountered: %d' % index)
    print('Total number of configs computed: %d' % analysed)
    print('Time passed: %f' % (time.time() - start))
