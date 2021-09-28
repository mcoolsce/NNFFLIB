import tensorflow as tf
import numpy as np
from .help_functions import load_xyz, XYZLogger
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))   

cell_list_op = tf.load_op_library(os.path.dirname(__file__) + '/cell_list_op.so')


class DataSet(object):
    def __init__(self, tfr_files, num_configs, cutoff = 5., batch_size = 16, test = False, num_shuffle = -1, float_type = 32, 
                 num_parallel_calls = 8, strategy = None, list_of_properties = ['positions', 'numbers', 'energy', 'rvec', 'forces'],
                 additional_property_definitions = {}):
        self.cutoff = cutoff
        self.num_configs = num_configs
        self.list_of_properties = list_of_properties
        self.batch_size = batch_size
        
        if not 'positions' in list_of_properties:
            raise RuntimeError('Positions are required.')
        if not 'numbers' in list_of_properties:
            raise RuntimeError('Numbers are required.')
        
        if float_type == 32:
            self.float_type = tf.float32
            self.zero = 0.        
        elif float_type == 64:
            self.float_type = tf.float64
            self.zero = np.float64(0.)
            
        # For every property, the tuple (shape, pad_index, dtype) is defined.
        self.property_definitions = {'energy': ([1], self.zero, self.float_type),
                                     'positions': ([None, 3], self.zero, self.float_type),
                                     'numbers': ([None], -1, tf.int32),
                                     'pairs': ([None, None, 4], -1, tf.int32),
                                     'rvec': ([3, 3], self.zero, self.float_type),
                                     'vtens' : ([3, 3], self.zero, self.float_type),
                                     'forces' : ([None, 3], self.zero, self.float_type)}
        self.property_definitions.update(additional_property_definitions)
        
        padding_shapes = {'pairs': [None, None, 4], 'rvec' : [3, 3]}
        padding_values = {'pairs': -1, 'rvec' : self.zero}
        for key in self.list_of_properties:
            if not key in self.property_definitions.keys():
                raise RuntimeError('The property definition of the key "%s" is not known. Use the additional_property_definitions keyword to update its definition.' % key)
            padding_shapes[key] = self.property_definitions[key][0]
            padding_values[key] = self.property_definitions[key][1]
        
        if num_shuffle == -1:
            num_shuffle = self.num_configs
        
        if test:
            print('Initializing test set')
        else:
            print('Initializing training set')
            
        print('Total number of systems found: %d' % self.num_configs)
        print('Initializing the dataset with cutoff radius %f' % self.cutoff)
        print('Using float%d' % float_type)
        print('Batch size: %d' % batch_size)
        print('List of properties: ' + str(self.list_of_properties))
        if not 'rvec' in self.list_of_properties:
            print('WARNING: internally rvec will be set 100 * np.eye(3) to calculate pairs') 
        print('')
        
        self._dataset = tf.data.TFRecordDataset(tfr_files)
        
        if test:
            self._dataset = self._dataset.repeat(1) # Repeat only once
        else:
            self._dataset = self._dataset.shuffle(num_shuffle)
            self._dataset = self._dataset.repeat() # Repeat indefinitely
         
        self._dataset = self._dataset.map(self.parser, num_parallel_calls = num_parallel_calls)
        self._dataset = self._dataset.padded_batch(self.batch_size, padded_shapes = padding_shapes, padding_values = padding_values)      
        self._dataset = self._dataset.prefetch(self.batch_size)
        
        if strategy:
            self._dataset = strategy.experimental_distribute_dataset(self._dataset)
    
    def parser(self, element):
        # Create the feature vector based on our property list
        feature = {}
        for key in self.list_of_properties:
            feature[key] = tf.io.FixedLenFeature((), tf.string)   
        parsed_features = tf.io.parse_single_example(element, features = feature)
        
        output_dict = {}
        for key in self.list_of_properties:
            output_dict[key] = tf.io.decode_raw(parsed_features[key], self.property_definitions[key][2])
            
        if not 'rvec' in self.list_of_properties:
            output_dict['rvec'] = 100. * tf.eye(3, dtype=self.float_type)
                
        # Replace every None with the number of atoms
        N = tf.shape(output_dict['numbers'])[0]
        for key in self.list_of_properties:
            new_shape = [N if dim is None else dim for dim in self.property_definitions[key][0]]
            output_dict[key] = tf.reshape(output_dict[key], new_shape)
        
        if self.float_type == tf.float64:
            pairs = cell_list_op.cell_list(tf.cast(output_dict['positions'], dtype = tf.float32), tf.cast(output_dict['rvec'], dtype = tf.float32), np.float32(self.cutoff))
        else:
            pairs = cell_list_op.cell_list(output_dict['positions'], output_dict['rvec'], np.float32(self.cutoff))
        
        output_dict['pairs'] = pairs
        return output_dict

def convert_np_value(value, float_converter):
    if type(value) == int:
        return np.int32(value)
    elif type(value) == float:
        return float_converter(value)
    elif type(value) == np.ndarray:
        if value.dtype == np.int:
            return np.int32(value)
        elif value.dtype == np.float:
            return float_converter(value)
        else:
            raise RuntimeError('Could not convert the value of dtype %s' % str(value.dtype))
    else:
        raise RuntimeError('Could not convert the value of type %s' % str(type(value)))
            
 
class TFRWriter(object):
    def __init__(self, filename, list_of_properties = ['positions', 'numbers', 'energy', 'rvec', 'forces'], float_type = 32,
                 verbose=True, reference = 0.):
        ''' Possible properties:
            positions ('pos' in xyz)
            numbers ('Z' in xyz)
            energy
            rvec ('Lattice' in xyz)
            forces ('force' in xyz)
            vtens
            other
        '''
        if float_type == 32:
            self.float_converter = np.float32
        elif float_type == 64:
            self.float_converter = np.float64
        
        self.filename = filename   
        self.tfr_file = tf.io.TFRecordWriter(self.filename)
        self.list_of_properties = list_of_properties
        self.num_configs = 0
        self.num_atoms = 0
        self.verbose = verbose
        self.stats = []
        self.reference = reference
        
        if self.verbose:
            print('Using float%d' % float_type)
            print('Storing the following properties: ' + str(self.list_of_properties))
            print('Reference energy: ' + str(self.reference))
        
    def write(self, **kwargs):
        kwargs['energy'] -= self.reference
    
        feature = {}
        to_store = self.list_of_properties.copy()
        for key, value in kwargs.items():
            if not key in self.list_of_properties:
                raise RuntimeError('Key %s does not appear in the list of properties' % key)
            to_store.remove(key)
            
            value = convert_np_value(value, self.float_converter)
            feature[key] = _bytes_feature(tf.compat.as_bytes(value.tostring()))
            
        if len(to_store) >= 1:
            raise RuntimeError('Missing properties: ' + str(to_store))   
        
        example = tf.train.Example(features = tf.train.Features(feature = feature))
        self.tfr_file.write(example.SerializeToString())

        self.num_configs += 1
        self.num_atoms += len(kwargs['numbers'])
        
        if 'energy' in self.list_of_properties:
            self.stats.append(kwargs['energy'])
        
        if self.num_configs % 1000 == 0 and self.verbose:
            print('Storing configuration %d' % self.num_configs) # Can be replaced by a simple progressbar
        
    def write_from_xyz(self, xyzfiles):
        if type(xyzfiles) == str:
            xyzfiles = [xyzfiles]
        
        for filename in xyzfiles:
            for data in load_xyz(filename):
                # Look at list_of_properties and find the corresponding key
                kwargs = {}
                for item in self.list_of_properties:
                    if item == 'positions':
                        kwargs['positions'] = data['pos']
                    elif item == 'numbers':
                        kwargs['numbers'] = data['Z']
                    elif item == 'rvec':
                        kwargs['rvec'] = data['Lattice']
                    elif item == 'forces':
                        kwargs['forces'] = data['force']
                    else:
                        kwargs[item] = data[item]
                        
                self.write(**kwargs)
    
    def close(self):
        self.tfr_file.close()
        
        if self.verbose:
            print('')
            print('%d configurations were written to file %s' % (self.num_configs, self.filename))
            print('In total, %d atoms were stored' % self.num_atoms)
            
            if 'energy' in self.list_of_properties:
                print('Mean energy: %f' % np.mean(self.stats))
                print('Std energy: %f' % np.std(self.stats))

                print('Max energy: %f' % np.max(self.stats))
                print('Min energy: %f' % np.min(self.stats))
            
            print('')   
