import tensorflow as tf
import numpy as np
from molmod.units import angstrom, electronvolt, pascal
from scipy.optimize import minimize
import pickle

import os
cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')


class Model(tf.Module):
    def __init__(self, cutoff, restore_file = None, float_type = 32, reference = 0, xla = False, per_atom_reference = 0.):
        super(Model, self).__init__()
         
        self.cutoff = cutoff
        self.restore_file = restore_file
        self.reference = reference
        self.per_atom_reference = per_atom_reference
        
        if float_type == 32:
            self.float_type = tf.float32
        elif float_type == 64:
            self.float_type = tf.float64
        else:
            raise RuntimeError('Float type %d not implemented.' % float_type)
            
        if xla:
            print('Enabling XLA')
            self.compute_properties = self.xla_compute_properties
        else:
            self.compute_properties = self.default_compute_properties
            
            
    @classmethod
    def from_restore_file(cls, restore_file, **kwargs):
        ''' kwargs may include:
                float_type (32 or 64)
                reference (default is 0.0)
                longrange_compute (default is None)
                xla (default is False)
        '''
        kwargs.update(pickle.load(open(restore_file + '.pickle', 'rb')))
        kwargs['restore_file'] = restore_file
        my_model = cls(**kwargs)
        my_model.load_checkpoint()
        return my_model
            
                   
    def load_checkpoint(self, ckpt = None):
        if self.restore_file is not None:
            if ckpt is None:
                ckpt = tf.train.Checkpoint(model = self)
            status = ckpt.restore(self.restore_file)
            try:
                status.assert_consumed() # Do some basis checks
                print('The model was successfully restored from file %s' % self.restore_file)
            except Exception as e:
                print(e)
        else:
            print('Starting from random parameters.')
                
    
    def calculate_reference_energy(self, numbers, masks):
        if not self.reference is None:
            return self.reference
        if not self.per_atom_reference is None:
            return tf.reduce_sum(masks['elements_mask'], [-1]) * self.per_atom_reference
        return 0.
            
    
    @tf.function(autograph = False, experimental_relax_shapes = True)
    def default_compute_properties(self, inputs, list_of_properties):
        return self._compute_properties(inputs, list_of_properties)
        
        
    @tf.function(autograph = False, experimental_relax_shapes = True, jit_compile = True)
    def xla_compute_properties(self, inputs, list_of_properties):
        return self._compute_properties(inputs, list_of_properties)
    
    
    def _compute_properties(self, inputs, list_of_properties):
        rvec = inputs['rvec']
        positions = inputs['positions']
        numbers = inputs['numbers']
        pairs = inputs['pairs']
        
        self.batches = tf.shape(pairs)[0]
        self.N = tf.shape(pairs)[1]
        self.J = tf.shape(pairs)[2]
        
        strain = tf.zeros([self.batches, 3, 3], dtype = self.float_type)
        
        with tf.GradientTape(persistent = True) as force_tape:
            force_tape.watch(strain)
            force_tape.watch(positions)
            
            positions += tf.linalg.matmul(positions, strain)
            rvec += tf.linalg.matmul(rvec, strain)
            
            masks = self.compute_masks(numbers, pairs)
            gather_center, gather_neighbor = self.make_gather_list(pairs, masks['neighbor_mask_int'])
            dcarts, dists = self.compute_distances(positions, numbers, rvec, pairs, masks['neighbor_mask_int'],
                                                   gather_center, gather_neighbor)
                                                                    
            energy, atomic_properties = self.internal_compute(dcarts, dists, numbers, masks, gather_neighbor)
        
        reference_energy = self.calculate_reference_energy(numbers, masks) 
        calculated_properties = {'energy' : energy + reference_energy}
        
        if 'forces' in list_of_properties:
            model_gradient = force_tape.gradient(energy, positions)
            calculated_properties['forces'] = -model_gradient   # These are not masked yet!
        
        if 'vtens' in list_of_properties or 'stress' in list_of_properties:
            vtens = force_tape.gradient(energy, strain)
            calculated_properties['vtens'] = vtens
            
        if 'masks' in list_of_properties:
            calculated_properties.update(masks)

        return calculated_properties
        
        
    @tf.function(autograph = False, experimental_relax_shapes = True)  
    def _compute_hessian(self, inputs, include_rvec = True):
        input_rvec = inputs['rvec']
        input_positions_tmp = inputs['positions']
        input_numbers = inputs['numbers']
        input_pairs = inputs['pairs']
        
        # Collecting all the necessary variables
        self.batches = tf.shape(input_pairs)[0]
        self.N = tf.shape(input_pairs)[1]
        if include_rvec:
            gvecs = tf.linalg.inv(input_rvec)
            fractional = tf.einsum('ijk,ikl->ijl', input_positions_tmp, gvecs)
            variables = [tf.reshape(fractional, [self.batches, -1]), tf.reshape(input_rvec, [self.batches, 9])]
        else:
            variables = [tf.reshape(input_positions_tmp, [self.batches, -1])]
        
        all_coords = tf.concat(variables, axis=1)
        F = tf.shape(all_coords)[1] # NUMBER OF DIMENSIONS OF THE HESSIAN
        all_coords = tf.stop_gradient(all_coords)
        
        # Extracting the variables
        if include_rvec:
            rvec = tf.reshape(all_coords[:, self.N*3:self.N*3+9], [self.batches, 3, 3]) 
            fractional = tf.reshape(all_coords[:, :self.N*3], [self.batches, -1, 3])
            positions = tf.einsum('ijk,ikl->ijl', fractional, rvec)
        else:
            positions = tf.reshape(all_coords[:, :self.N*3], [self.batches, -1, 3])
            rvec = tf.expand_dims(tf.eye(3, dtype=self.float_type), [0]) * 100.
        
        # The shortrange contributions   
        masks = self.compute_masks(input_numbers, input_pairs)
        gather_center, gather_neighbor = self.make_gather_list(input_pairs, masks['neighbor_mask_int'])
        dcarts, dists = self.compute_distances(positions, input_numbers, rvec, input_pairs, masks['neighbor_mask_int'],
                                               gather_center, gather_neighbor)
                                                               
        energy, atomic_properties = self.internal_compute(dcarts, dists, input_numbers, masks, gather_neighbor)
        
        # No need to include reference energies here 
        hessian = tf.hessians(energy, all_coords)[0]
        hessian = tf.reshape(hessian, [F, F])
        return hessian
        
    
    def make_gather_list(self, pairs, neighbor_mask_int):
        self.J = tf.shape(pairs)[2]
        batch_indices = tf.tile(tf.reshape(tf.range(self.batches), [-1, 1, 1, 1]), [1, self.N, self.J, 1])
        index_center = tf.tile(tf.reshape(tf.range(self.N), [1, -1, 1, 1]), [self.batches, 1, self.J, 1])
        
        # A tensor [batches, N, J, 2] containing the indices of the center atoms
        gather_center = tf.concat([batch_indices, index_center], axis = 3)
        # A tensor [batches, N, J, 2] containing the indices of the neighbors neighbors
        gather_neighbor = tf.concat([batch_indices, tf.expand_dims(pairs[:, :, :, 0], [-1])], axis = 3)

        # Gathering on index -1 raises some errors on CPU when checking bounds
        gather_neighbor *= tf.expand_dims(neighbor_mask_int, [-1]) 
        
        return gather_center, gather_neighbor
        
    
    def compute_masks(self, numbers, pairs):
        neighbor_mask_int = tf.cast(tf.not_equal(pairs[:, :, :, 0], -1), tf.int32)          # shape [batches, N, J]
        neighbor_mask = tf.cast(tf.not_equal(pairs[:, :, :, 0], -1), self.float_type)       # shape [batches, N, J]
        elements_mask = tf.cast(numbers > 0, self.float_type)                               # shape [batches, N]
        return {'neighbor_mask_int' : neighbor_mask_int, 'neighbor_mask' : neighbor_mask, 'elements_mask' : elements_mask}

                
    def compute_distances(self, positions, numbers, rvecs, pairs, neighbor_mask_int, gather_center, gather_neighbor):     
        # Making an [batches, N, 3, 3] rvecs
        rvecs_matmul = tf.tile(tf.expand_dims(rvecs, [1]), [1, self.N, 1, 1])
        
        # Computing the relative vectors for each pair
        dcarts = tf.add(
            tf.subtract(
                tf.gather_nd(positions, gather_neighbor),
                tf.gather_nd(positions, gather_center)
            ), tf.matmul(tf.cast(pairs[:, :, :, 1:], dtype = self.float_type), rvecs_matmul))

        # Avoid dividing by zero when calculating derivatives
        zero_division_mask = tf.cast(1 - neighbor_mask_int, self.float_type)
        dcarts += tf.expand_dims(zero_division_mask, [-1])
        
        # Computing the squared distances
        dists = tf.sqrt(tf.reduce_sum(tf.square(dcarts), [-1]) + 1e-20)

        return dcarts, dists
     
      
    def save(self, output_file):
        raise NotImplementedError
        
        
    def preprocess(self, positions, numbers, rvec):
        # First we convert the numpy arrays into real tensor with the correct data type
        tf_rvec = tf.convert_to_tensor(rvec, dtype = self.float_type)
        tf_positions = tf.convert_to_tensor(positions, dtype = self.float_type)
        tf_numbers = tf.convert_to_tensor(numbers, dtype = tf.int32)
        if self.float_type == tf.float64:
            tf_pairs = cell_list_op.cell_list(tf.cast(tf_positions, dtype = tf.float32), tf.cast(tf_rvec, dtype = tf.float32), np.float32(self.cutoff))
        else:
            tf_pairs = cell_list_op.cell_list(tf_positions, tf_rvec, np.float32(self.cutoff))   
        # Pad each input
        inputs = {'numbers' : tf.expand_dims(tf_numbers, [0]), 'positions' : tf.expand_dims(tf_positions, [0]),
                  'rvec' : tf.expand_dims(tf_rvec, [0]), 'pairs' : tf.expand_dims(tf_pairs, [0])}
        return inputs
        
        
    def compute(self, positions, numbers, rvec = 100 * np.eye(3), list_of_properties = ['energy', 'forces']):
        ''' Returns the energy and forces'''
        inputs = self.preprocess(positions, numbers, rvec)
        tf_calculated_properties = self.compute_properties(inputs, list_of_properties)
        
        calculated_properties = {}
        for key in tf_calculated_properties.keys():
            calculated_properties[key] = tf_calculated_properties[key].numpy()[0]
        
        if 'stress' in list_of_properties:   
            calculated_properties['stress'] = - calculated_properties['vtens'] / np.linalg.det(rvec) * (electronvolt / angstrom**3) / (1e+09 * pascal) # in GPa

        return calculated_properties
        
    
    def compute_hessian(self, positions, numbers, rvec = 100 * np.eye(3), include_rvec = True):
        ''' Returns the energy and forces'''
        inputs = self.preprocess(positions, numbers, rvec)
        tf_hessian = self._compute_hessian(inputs, include_rvec = include_rvec)
        return tf_hessian.numpy()
        
   
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):
        raise NotImplementedError
