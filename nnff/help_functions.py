import tensorflow as tf
import numpy as np
from molmod.periodic import periodic
import time
import shlex

def weight_variable(shape, name, trainable = True, stddev = None):
    if stddev is None:
        initial = tf.random.truncated_normal(shape, stddev = 4. / np.sqrt(shape[-1]))
    else:
        initial = tf.random.truncated_normal(shape, stddev = stddev)
    
    return tf.Variable(initial, name = name, trainable = trainable)


def bias_variable(shape, name, initial_value = 0.0, trainable = True):
    initial = tf.constant(initial_value, shape=shape)

    return tf.Variable(initial, name = name, trainable = trainable)
    

def make_comment(energy, rvecs = None, forces = None, vtens = None):
    comment = 'Properties=species:S:1:pos:R:3:Z:I:1'
    if not forces is None:
        comment += ':force:R:3'     
    if rvecs is not None:
        comment += ' Lattice="%f %f %f %f %f %f %f %f %f"' % tuple(rvecs.flatten())
    if vtens is not None:
        comment += ' vtens="%f %f %f %f %f %f %f %f %f"' % tuple(vtens.flatten())
    comment += ' energy=' + str(energy)
    return comment
    

class XYZLogger(object):
    def __init__(self, filename):
        self.filename = filename
        file = open(self.filename, 'w')
        file.close()
        
    def write(self, energy, rvec, numbers, positions, forces = None, vtens = None):
        self.file = open(self.filename, 'a')
        
        N = np.shape(positions)[0]

        self.file.write('%d\n' % N)
        self.file.write(make_comment(energy, rvec, forces = forces, vtens = vtens) + '\n')
        
        for atom in range(N):
            symbol = periodic[numbers[atom]].symbol
            if forces is None:
                newline = '%s\t%s\t%s\t%s\t%d' % (symbol, str(positions[atom, 0]), str(positions[atom, 1]), str(positions[atom, 2]), numbers[atom])
            else:
                fx, fy, fz = tuple(forces[atom])
                newline = '%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' % (symbol, str(positions[atom, 0]), str(positions[atom, 1]), str(positions[atom, 2]), numbers[atom], str(fx), str(fy), str(fz))
            self.file.write(newline + '\n')
        self.file.close()


def convert(value):
    splits = value.split()
    if len(splits) == 1:
        try:
            return int(value)
        except:
            try:
                return float(value)
            except:
                return value
    else:
        try:
            return np.array(splits, dtype=np.int)
        except:
            try:
                return np.array(splits, dtype=np.float)
            except:
                return np.array(splits, dtype=np.str)
            

def parse_comment(comment):
    ''' simple keyword parser '''
    parsed_comment = {}
    splits = shlex.split(comment)
    for split in splits:
        keyword, value = split.split('=')     
        parsed_comment[keyword] = convert(value)      
    return parsed_comment


def make_converter(symbol):
    if symbol == 'S':
        return np.str
    elif symbol == 'I':
        return np.int
    elif symbol == 'R':
        return np.float


def read_properties(properties):
    splits = properties.split(':')
    names = splits[::3]
    types = splits[1::3]
    lengths = splits[2::3]
    keys = []
    for index, name in enumerate(names):
        keys.append((name, int(lengths[index]), make_converter(types[index]), []))
    return keys


def load_xyz(xyz_file):   
    with open(xyz_file) as inputfile:
        while True:
            line = inputfile.readline()
            if not line: # End of File
                break
            
            N_atoms = int(line)
            comment = inputfile.readline() # Reading the comment
            data = parse_comment(comment)
            
            # Reading the properties
            keys = read_properties(data['Properties'])

            for atom in range(N_atoms):
                line = inputfile.readline()[:-1]
                splits = line.split()
                
                pointer = 0
                for key in keys:
                    key[3].append(splits[pointer:pointer + key[1]])
                    pointer += key[1]
            
            for key in keys:
                data[key[0]] = np.array(key[3], dtype=key[2])
                if key[1] == 1:
                    data[key[0]] = data[key[0]].flatten()
            
            if not 'Z' in data.keys(): # Symbols -> numbers
                data['Z'] = np.zeros(len(data['species']), dtype=np.int)
                for index, symbol in enumerate(data['species']):
                    data['Z'][index] = periodic[symbol].number
                         
            yield data
            

class ProgressBar(object):
    def __init__(self, total_steps, verbose = False):
        self.total_steps = total_steps
        self.counter = 0
        self.epoch = 0
        self.train_time = 0
        self.verbose = verbose
    
    def start_training(self):
        self.train_start = time.time()
        
    def step(self):
        self.counter += 1
        
    def start_validating(self):
        self.validation_start = time.time()
        self.validation_time = 0
        
    def draw(self, validating = False):
        if not self.verbose:
            return
        percentage = '%5.2f' % (100. * self.counter / self.total_steps)
        
        filled = int(50 * self.counter / self.total_steps)
        bar = '█' * filled + '-' * (50 - filled)
        
        self.prefix = 'Training epoch %d' % self.epoch
        suffix = '[ %d/%d  ' % (self.counter, int(self.total_steps))
        
        now = time.time()
        
        if not validating:
            self.train_time += now - self.train_start
            self.train_start = now
        
        if self.counter != 0:    
            estimated_time = (self.total_steps - self.counter) / self.counter * self.train_time
        else:
            estimated_time = 0
            
        suffix += time.strftime("%H:%M:%S", time.gmtime(int(self.train_time))) + ' < '
        suffix += time.strftime("%H:%M:%S", time.gmtime(int(estimated_time))) + ' ]'
        
        if validating:
            self.validation_time += now - self.validation_start
            self.validation_start = now
            
            if (self.validation_time * 2) % 2 < 1: 
                suffix += ' Validating'
            else:
                suffix += ' ' * 11

        print('\r%s |%s| %s%% %s' % (self.prefix, bar, percentage, suffix), end = '\r')
        if self.counter >= self.total_steps:
            self.epoch += 1
            self.counter -= self.total_steps
            self.train_time = 0
            
            
class ScreenLog(object):
    def __init__(self, validation_losses): 
        self.blank_header = '{:^10s}{:^20s}'
        self.blank = '{:^10.3f}{:^20f}'
        self.values = ['epoch', 'training loss']
        
        for loss in validation_losses:
            title = loss.get_title()
            max_size = max(len(title), 15)
            self.blank_header += '{:^%ds}' % max_size
            self.blank += '{:^%df}' % max_size
            self.values.append(title)
            
        self.blank_header += '{:^15s}{:^15s}'
        self.blank += '{:^15f}{:^15f}'
        self.values += ['time_passed', 'total_time']
        
        self.header = self.blank_header.format(*self.values)
       
    def print_header(self):
        print(self.header)
        print('-' * len(self.header))
    
    def print_losses(self, losses, epoch = 0, average_training_loss = 0, time_passed = 0, total_time = 0):
        values = [epoch, average_training_loss] + list(losses) + [time_passed, total_time]
        print((len(self.blank.format(*values)) + 20) * ' ', end = '\r')
        print(self.blank.format(*values))  
        
        
        
        
        
