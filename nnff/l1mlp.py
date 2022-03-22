import tensorflow as tf
import numpy as np
from nnff.model import Model
from nnff.help_functions import bias_variable
import pickle
from molmod.units import angstrom


def weight_variable(shape, name, trainable = True):
    # Xavier uniform initialization, just like schnetpack
    assert len(shape) == 2
    bound = np.sqrt(6. / np.sum(shape))
    initial = tf.random.uniform(shape, minval = -bound, maxval = bound)
    return tf.Variable(initial, name = name, trainable = trainable)


def activation(tensor):
    return tf.math.softplus(tensor) - np.log(2.0)
    

def f_cutoff(input_tensor, cutoff, float_type = tf.float32):
    cutoffs = 0.5 * (tf.cos(input_tensor * np.pi / cutoff) + 1.0)
    return tf.where(input_tensor > cutoff, tf.zeros(tf.shape(input_tensor), dtype=float_type), cutoffs)
    
    
class SelfInteraction(tf.Module):
    def __init__(self, dim_in, dim_out, name, use_activation = True, use_bias = True, float_type = tf.float32):
        super(SelfInteraction, self).__init__()
        self.float_type = float_type
        self.use_activation = use_activation
        self.use_bias = use_bias
        self.weights = weight_variable([dim_in, dim_out], name + '_weights')
        if self.use_bias:
            self.bias = bias_variable([dim_out], name + '_bias')
    
    def __call__(self, input_features, mask = None):
        if self.float_type == tf.float64:
            weights = tf.cast(self.weights, dtype=tf.float64)
        else:
            weights = self.weights
                
        features = tf.tensordot(input_features, weights, [[-1], [0]]) 
        if self.use_bias:
            if self.float_type == tf.float64:
                bias = tf.cast(self.bias, dtype=tf.float64)
            else:
                bias = self.bias
            features += bias
        if self.use_activation:
            features = activation(features)
        if not mask is None:
            features *= mask
            
        return features
        

class NormActivation(tf.Module):
    def __init__(self, axis = -1, float_type = tf.float32):
        super(NormActivation, self).__init__()
        self.axis = axis
        
    def __call__(self, inputs):
        norms = tf.sqrt(1e-10 + tf.reduce_sum(inputs**2, self.axis))
        return inputs * tf.expand_dims(activation(norms) / norms, self.axis)
    

class InteractionBlock(tf.Module):
    def __init__(self, layer_index, num_features, num_filters, n_max, float_type = tf.float32):
        super(InteractionBlock, self).__init__()
        
        self.float_type = float_type
        self.num_filters = num_filters
        self.layer_index = layer_index
        
        self.scalar_interaction1 = SelfInteraction(num_features, num_filters, 'interaction_%d_scalar_interaction1' % layer_index, use_activation = False, float_type = float_type)

        # The radial_filter_ijk with i the angular moment of the atomic feature and j the angular moment of the distance representation
        # and k the output dimension
        self.radial_filter_000 = RadialFilter(n_max, num_filters, name = 'interaction_%d_radial_filter_000' % layer_index, float_type = float_type)
        self.radial_filter_011 = RadialFilter(n_max, num_filters, name = 'interaction_%d_radial_filter_011' % layer_index, float_type = float_type)
        
        if self.layer_index >= 1:
            self.vector_interaction1 = SelfInteraction(num_features, num_filters, 'interaction_%d_vector_interaction1' % layer_index, use_activation = False, use_bias = False, float_type = float_type)
            self.radial_filter_101 = RadialFilter(n_max, num_filters, name = 'interaction_%d_radial_filter_101' % layer_index, float_type = float_type)
            self.radial_filter_110 = RadialFilter(n_max, num_filters, name = 'interaction_%d_radial_filter_110' % layer_index, float_type = float_type)
            self.radial_filter_111 = RadialFilter(n_max, num_filters, name = 'interaction_%d_radial_filter_111' % layer_index, float_type = float_type)
        
            self.scalar_interaction2 = SelfInteraction(2 * num_filters, num_features, 'interaction_%d_scalar_interaction2' % layer_index, use_activation = True, float_type = float_type)
            self.vector_interaction2 = SelfInteraction(3 * num_filters, num_features, 'interaction_%d_vector_interaction2' % layer_index, use_activation = False, use_bias = False, float_type = float_type)
        else:
            self.scalar_interaction2 = SelfInteraction(num_filters, num_features, 'interaction_%d_scalar_interaction2' % layer_index, use_activation = True, float_type = float_type)
            self.vector_interaction2 = SelfInteraction(num_filters, num_features, 'interaction_%d_vector_interaction2' % layer_index, use_activation = False, use_bias = False, float_type = float_type)
        
        self.norm_activation = NormActivation(axis = [2])
        
        self.scalar_interaction3 = SelfInteraction(num_features, num_features, 'interaction_%d_scalar_interaction3' % layer_index, use_activation = False, float_type = float_type)
        self.vector_interaction3 = SelfInteraction(num_features, num_features, 'interaction_%d_vector_interaction3' % layer_index, use_activation = False, use_bias = False, float_type = float_type)
        
    
    def __call__(self, scalar_input, vector_input, radial_features, angles, elements_mask, neighbor_mask, smooth_cutoff_mask, gather_neighbor):       
        scalar_message = self.scalar_interaction1(scalar_input)
        scalar_neighbors = tf.gather_nd(scalar_message, gather_neighbor) # [batches, N, J, F]

        W000 = self.radial_filter_000(radial_features) * smooth_cutoff_mask # [batches, N, J, F]
        W011 = self.radial_filter_011(radial_features) * smooth_cutoff_mask

        # angles has a shape of [batches, N, J, 3]
        tensorproduct_000 = tf.reduce_sum(scalar_neighbors * W000, axis = [2]) # [batches, N, F]
        tensorproduct_011 = tf.reduce_sum(tf.expand_dims(scalar_neighbors, [3]) * tf.expand_dims(W011, [3]) * tf.expand_dims(angles, [-1]), axis = [2]) # [batches, N, 3, F]

        if self.layer_index >= 1:
            vector_message = self.vector_interaction1(vector_input)
            vector_neighbors = tf.gather_nd(vector_message, gather_neighbor) # [batches, N, J, 3, F]
            
            W101 = self.radial_filter_101(radial_features) * smooth_cutoff_mask
            W110 = self.radial_filter_110(radial_features) * smooth_cutoff_mask
            W111 = self.radial_filter_111(radial_features) * smooth_cutoff_mask
        
            tensorproduct_101 = tf.reduce_sum(vector_neighbors * tf.expand_dims(W101, [3]), axis = [2]) # [batches, N, 3, F]
            
            dotproduct = tf.reduce_sum(vector_neighbors * tf.expand_dims(angles, [-1]), [3]) # [batches, N, J, F]
            tensorproduct_110 = tf.reduce_sum(dotproduct * W110, axis = [2]) # [batches, N, F]
            
            crossproduct = tf.linalg.cross(tf.transpose(vector_neighbors, [0, 1, 2, 4, 3]), tf.tile(tf.expand_dims(angles, [3]), [1, 1, 1, self.num_filters, 1]))
            tensorproduct_111 = tf.reduce_sum(tf.transpose(crossproduct, [0, 1, 2, 4, 3]) * tf.expand_dims(W111, [3]), axis = [2]) # [batches, N, 3, F]
            
            # Concatenating the tensorproduct with simular angular momenta
            scalar_message = tf.concat([tensorproduct_000, tensorproduct_110], axis = 2)
            vector_message = tf.concat([tensorproduct_011, tensorproduct_101, tensorproduct_111], axis = 3)
        else:
            scalar_message = tensorproduct_000
            vector_message = tensorproduct_011
      
        scalar_message = self.scalar_interaction2(scalar_message)
        vector_message = self.norm_activation(self.vector_interaction2(vector_message))

        scalar_message = self.scalar_interaction3(scalar_message)
        vector_message = self.vector_interaction3(vector_message)

        return scalar_message, vector_message
        
        
class RadialFilter(tf.Module):
    def __init__(self, n_max, num_filters, name, float_type = tf.float32):
        super(RadialFilter, self).__init__()
        self.interaction1 = SelfInteraction(n_max, num_filters, name + '_layer1', use_activation = True, float_type = float_type)
        self.interaction2 = SelfInteraction(num_filters, num_filters, name + '_layer2', use_activation = False, float_type = float_type)

              
    def __call__(self, radial_features):
        return self.interaction2(self.interaction1(radial_features))
        
        
class OutputLayer(tf.Module):
    def __init__(self, prefix_name, layer_sizes, initial_size, float_type = tf.float32):
        super(OutputLayer, self).__init__()
        self.float_type = float_type
        
        previous_size = initial_size
        self.layer_sizes = layer_sizes
        
        self.output_weights = []
        self.output_biases = []
        
        for i, layer_size in enumerate(self.layer_sizes):    
            self.output_weights.append(weight_variable([previous_size, layer_size], prefix_name + '_weights_layer_%d' % i))
            self.output_biases.append(bias_variable([1, 1, layer_size], prefix_name + '_bias_layer_%d' % i))
            
            previous_size = layer_size
                  
    def __call__(self, atom_features):
        for i, layer_size in enumerate(self.layer_sizes):
            w = self.output_weights[i]
            b = self.output_biases[i] 
            
            if self.float_type == tf.float64:
                w = tf.cast(w, dtype = tf.float64)
                b = tf.cast(b, dtype = tf.float64)           
            
            atom_features = tf.tensordot(atom_features, w, [[2], [0]]) + b
            if i < len(self.layer_sizes) - 1: # No activation function for the final layer
                atom_features = activation(atom_features)
                        
        return atom_features


class L1MLP(Model):
    def __init__(self, cutoff = 5., n_max = 25, num_features = 64, start = 0.0, num_layers = 3, end = None, num_filters = -1, **kwargs):
        ''' L1MLP model
            The architecture is based on the NequIP (https://arxiv.org/abs/2101.03164)
        '''
        Model.__init__(self, cutoff, **kwargs) 
        if end is None:
            self.end = cutoff
        else:
            self.end = end
        self.n_max = n_max
        self.start = start

        if num_filters == -1:
            self.num_filters = H
        else:
            self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_features = num_features # Size of the hidden state vector
        
        self.offsets = tf.cast(tf.linspace(self.start, self.end, self.n_max), self.float_type)
        self.widths = self.offsets[1] - self.offsets[0]
        self.offsets = tf.reshape(self.offsets, [1, 1, 1, -1])

        # Here, all variables are listed and other submodules
        self.init_features = tf.Variable(tf.random.normal([100, self.num_features], stddev = 1.), name = 'init_hidden_vector')
        
        self.interaction_blocks = []
        for layer_index in range(num_layers):
            self.interaction_blocks.append(InteractionBlock(layer_index, self.num_features, self.num_filters, self.n_max, float_type = self.float_type))
            
        self.output_layer = OutputLayer('output_layer', [int(self.num_features / 2), int(self.num_features / 4), 1], self.num_features, float_type = self.float_type)

       
    def save(self, output_file):
        data = {'cutoff' : self.cutoff, 'n_max' : self.n_max, 'start' : self.start, 'end' : self.end,
                'num_features' : self.num_features, 'num_layers' : self.num_layers, 'num_filters' : self.num_filters}
        pickle.dump(data, open(output_file + '.pickle', 'wb'))
        
        
    def internal_compute(self, dcarts, dists, numbers, masks, gather_neighbor):
        ''' Defining the radial features ''' 
        radial_features = tf.exp(- 0.5 * (tf.expand_dims(dists, [-1]) - self.offsets)**2 / self.widths**2) # shape [batches, N, J, n]
        
        smooth_cutoff_mask = f_cutoff(dists, cutoff = self.cutoff, float_type = self.float_type) # shape [batches, N, J]
        smooth_cutoff_mask = tf.expand_dims(smooth_cutoff_mask * masks['neighbor_mask'], [-1]) # shape [batches, N, J, 1]
        
        angles = dcarts / tf.expand_dims(dists, [-1])
            
        if self.float_type == tf.float64:
            init_features = tf.cast(self.init_features, dtype = tf.float64)
        else:
            init_features = self.init_features

        scalar_features = tf.nn.embedding_lookup(init_features, numbers * tf.cast(numbers > 0, tf.int32)) * tf.expand_dims(masks['elements_mask'], [-1])
        vector_features = tf.zeros([self.batches, self.N, 3, self.num_features], dtype = self.float_type)
        
        ''' The interaction layers '''
        for i in range(self.num_layers):
            scalar_message, vector_message = self.interaction_blocks[i](scalar_features, vector_features, radial_features, angles, masks['elements_mask'], masks['neighbor_mask'], smooth_cutoff_mask, gather_neighbor)
            scalar_features += scalar_message
            vector_features += vector_message
            
        scalar_features += tf.reduce_sum(vector_message**2, axis = [2]) # simple fix to avoid non existing gradients in the last layer

        atomic_energies = tf.reshape(self.output_layer(scalar_features), [self.batches, self.N])

        ''' The final energy '''
        energy = tf.reduce_sum(atomic_energies * masks['elements_mask'], [-1])

        return energy, atomic_energies
