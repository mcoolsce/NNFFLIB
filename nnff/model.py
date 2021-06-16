import tensorflow as tf
import numpy as np
from molmod.units import angstrom, electronvolt
from scipy.optimize import minimize
import pickle

import os
cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')

    
@tf.function(autograph = False, experimental_relax_shapes = True)
def single_pass_hessian(rvec, pairs, positions, numbers, model):
    dcarts, dists, mask_library, gather_neighbor = model.compute_distances(positions, numbers, rvec, pairs)
    mask_library = model.compute_masks(numbers, dists, model.cutoff, model.cutoff_transition_width, mask_library)
        
    energy, _ = model.internal_compute(dists, dcarts, numbers, mask_library, gather_neighbor)

    hessian = tf.hessians(energy, positions)[0]
    
    # Filtering out the position gradients
    #position_gradients = model_gradient * tf.expand_dims(mask_library[2], [-1])

    return hessian
   

class Model(tf.Module):
    def __init__(self, cutoff, restore_file = None, float_type = 32, do_ewald = False, reference = 0):
        super(Model, self).__init__()
         
        self.cutoff = cutoff
        self.restore_file = restore_file
        self.do_ewald = do_ewald
        assert not self.do_ewald
        
        self.reference = reference
        
        if float_type == 32:
            self.float_type = tf.float32
        elif float_type == 64:
            self.float_type = tf.float64
        else:
            raise RuntimeError('Float type %d not implemented.' % float_type)
            
            
    @classmethod
    def from_restore_file(cls, restore_file, float_type = 32, reference = None):
        data = pickle.load(open(restore_file + '.pickle', 'rb'))
        data['restore_file'] = restore_file
        data['float_type'] = float_type
        data['reference'] = reference
        my_model = cls(**data)
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
                
    
    def calculate_reference_energy(self, numbers):
        if not self.reference is None:
            return self.reference
        else:
            return 0.
        
    
    @tf.function(autograph = False, experimental_relax_shapes = True)
    def compute_properties(self, inputs, list_of_properties):
        rvec = inputs['rvec']
        input_positions = inputs['positions']
        numbers = inputs['numbers']
        pairs = inputs['pairs']
        
        self.batches = tf.shape(pairs)[0]
        self.N = tf.shape(pairs)[1]
        self.J = tf.shape(pairs)[2]
        
        gvecs = tf.linalg.inv(rvec)
        fractional = tf.einsum('ijk,ikl->ijl', input_positions, gvecs)
        
        with tf.GradientTape(persistent = True) as force_tape:
            force_tape.watch(rvec)
            positions = tf.einsum('ijk,ikl->ijl', fractional, rvec)
            force_tape.watch(positions)
            
            masks = self.compute_masks(numbers, pairs)
            gather_center, gather_neighbor = self.make_gather_list(pairs, masks['neighbor_mask_int'])
            dcarts, dists = self.compute_distances(positions, numbers, rvec, pairs, masks['neighbor_mask_int'],
                                                   gather_center, gather_neighbor)
                                                                    
            energy, atomic_properties = self.internal_compute(dcarts, dists, numbers, masks, gather_neighbor)
                 
            if self.do_ewald:
                raise NotImplementedError
                #energy += model.long_range_compute(charges, dists, positions, rvec, numbers, mask_library, gather_neighbor)
        
        reference_energy = self.calculate_reference_energy(numbers) 
        calculated_properties = {'energy' : energy + reference_energy}
        
        if 'forces' in list_of_properties:
            model_gradient = force_tape.gradient(energy, positions)
            calculated_properties['forces'] = -model_gradient   # These are not masked yet!
        
        if 'vtens' in list_of_properties:
            de_dc = force_tape.gradient(energy, rvec)
            vtens = tf.einsum('ijk,ijl->ikl', rvec, de_dc)
            calculated_properties['vtens'] = vtens
            
        if 'masks' in list_of_properties:
            calculated_properties.update(masks)
            
            #pair_force = force_tape.gradient(energy, dcarts)
            #matrix = tf.expand_dims(dcarts, [-1]) * tf.expand_dims(pair_force, [3])
            #vtens = tf.reduce_sum(matrix * tf.reshape(mask_library[0], [model.batches, model.N, model.J, 1, 1]), [1, 2])

        return calculated_properties
        
    
    def make_gather_list(self, pairs, neighbor_mask_int):
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

        return calculated_properties
        
   
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):
        raise NotImplementedError
