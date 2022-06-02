import tensorflow as tf
import numpy as np


class AbstractError(object):
    def __init__(self, my_key, scale_factor, per_atom):
        self.my_key = my_key
        self.scale_factor = scale_factor
        self.per_atom = per_atom
        if self.per_atom:
            assert self.my_key == 'energy' # Just a simple check for now
        
    def __call__(self, predicted_dict, target_dict):
        my_target = target_dict[self.my_key]
        my_prediction = predicted_dict[self.my_key]
        
        if self.my_key == 'energy':
            my_target = tf.reshape(my_target, [-1])
            
        if self.my_key == 'forces':
            sum_elements = tf.reduce_sum(predicted_dict['elements_mask']) * 3 # 3 force components per atom
            my_mask = tf.expand_dims(predicted_dict['elements_mask'], [-1])
        else:
            sum_elements = tf.cast(tf.math.reduce_prod(tf.shape(my_target)), my_target.dtype)
            my_mask = 1.0
            
        if self.per_atom:
            my_target /= tf.reduce_sum(predicted_dict['elements_mask'], [1])
            my_prediction /= tf.reduce_sum(predicted_dict['elements_mask'], [1])

        sum_loss = self._call(my_prediction * my_mask, my_target * my_mask) * self.scale_factor
        
        return sum_loss, sum_elements
        
    def _call(self, output, data):
        raise NotImplementedError
        
    
    def get_title(self):
        return '%s %s' % (self._title, self.my_key)
        


class MSE(AbstractError):
    def __init__(self, my_key, scale_factor = 1., per_atom = False):
        super(MSE, self).__init__(my_key, scale_factor, per_atom)
        self._title = 'MSE'
        
    def _call(self, my_prediction, my_target):
        return tf.reduce_sum((my_prediction - my_target)**2)
        
        
class MAE(AbstractError):
    def __init__(self, my_key, scale_factor = 1., per_atom = False):
        super(MAE, self).__init__(my_key, scale_factor, per_atom)
        self._title = 'MAE'
    
    def _call(self, my_prediction, my_target):
        return tf.reduce_sum(tf.abs(my_prediction - my_target))
