import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from nnff.training import Trainer
from nnff.learning_rate_manager import ExponentialDecayLearningRate
from nnff.schnet import SchNet
from nnff.datasets import DataSet, TFRWriter
from nnff.losses import MSE, MAE
from nnff.hooks import SaveHook
from glob import glob
import tensorflow as tf

if __name__ == '__main__:
    list_of_properties = ['positions', 'numbers', 'energy', 'rvec', 'forces']

    # Generate the tfr datasets from xyz-files
    if False:
        writer = TFRWriter('validation.tfr', list_of_properties = list_of_properties)
        writer.write_from_xyz(glob('trainfiles/*.xyz'))
        writer.close()
    if False:
        writer = TFRWriter('train.tfr', list_of_properties = list_of_properties)
        writer.write_from_xyz(glob('testfiles/*.xyz'))
        writer.close()

    # Choose your strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        train_data = DataSet(['train.tfr'], num_configs = 399902, cutoff = 5.0, batch_size = 64, float_type = 32, num_parallel_calls = 8, strategy = strategy, list_of_properties = list_of_properties)
        validation_data = DataSet(['validation.tfr'], num_configs = 3997, cutoff = 5.0, batch_size = 64, float_type = 32, num_parallel_calls = 8, strategy = strategy, list_of_properties = list_of_properties, test = True)
        
        # Initialize a new model ... 
        model = SchNet(cutoff = 5., n_max = 32, num_layers = 4, start = 0.0, end = 5.0, num_filters = 64, num_features = 512, shared_W_interactions = False, float_type = 32, cutoff_transition_width = 1.0, reference = 0.)     
        
        # ... or resume from an existing one.  
        #model = SchNet.from_restore_file('model_dir/model_name', reference = 0.)
          
        optimizer = tf.optimizers.Adam(3e-04)
        learning_rate_manager = ExponentialDecayLearningRate(initial_learning_rate = 3e-04, decay_rate = 0.5, decay_epochs = 60)

        losses = [MSE('energy', scale_factor = 1., per_atom = True), MSE('forces', scale_factor = 1.)]
        validation_losses = [MAE('energy', per_atom = True), MAE('forces', scale_factor = 1.)]
           
        savehook = SaveHook(model, ckpt_name = 'model_dir/model_name', max_to_keep = 5, save_period = 0.1, history_period = 4.0,
                            npz_file = 'model_dir/model_name.npz')

        trainer = Trainer(model, losses, train_data, validation_data, strategy = strategy, optimizer = optimizer, savehook = savehook, 
                          learning_rate_manager = learning_rate_manager, validation_losses = validation_losses)
        trainer.train(verbose = True, validate_first = True)
