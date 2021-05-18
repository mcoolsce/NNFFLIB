import tensorflow as tf
from molmod.units import angstrom
import numpy as np

def real_space_part(alpha, charges, neighbor_charges, dists, feature_cutoff, neighbor_mask, elements_mask, radial_mask, float_type = tf.float32):
    Z_matrix = tf.expand_dims(charges, [2]) * neighbor_charges * neighbor_mask * tf.expand_dims(elements_mask, [-1])

    dist_function = tf.math.erfc(alpha * dists * angstrom) / dists / angstrom
    screened_matrix = dist_function * Z_matrix
    
    return 0.5 * tf.reduce_sum(screened_matrix * radial_mask,  [1, 2])
    
def self_correction_part(alpha, charges, elements_mask):
    return alpha / np.sqrt(np.pi) * tf.reduce_sum(charges**2 * elements_mask, [1])
    
def generate_kvecs(gvecs, rvecs, gcut, float_type = tf.float32):
    gspacings = 1. / tf.sqrt(tf.reduce_sum(rvecs**2, [1])) # [batches, 3]
    gmax = tf.cast(tf.math.ceil(gcut / (tf.reduce_min(gspacings) / angstrom) - 0.5), tf.int32) # A number, the minimum of all batches

    gx = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [-1, 1, 1, 1]), [1, 2 * gmax + 1, 2 * gmax + 1, 1])
    gy = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [1, -1, 1, 1]), [2 * gmax + 1, 1, 2 * gmax + 1, 1])
    gz = tf.tile(tf.reshape(tf.range(-gmax, gmax + 1), [1, 1, -1, 1]), [2 * gmax + 1, 2 * gmax + 1, 1, 1])
    
    n = tf.reshape(tf.concat((gx, gy, gz), 3), [(2 * gmax + 1)**3, 3]) # [K, 3]
    k_vecs = 2 * np.pi * tf.einsum('ijk,lk->ilj', gvecs, tf.cast(n, dtype = float_type)) # [batches, K, 3]
    
    # Constructing the mask
    k2 = tf.reduce_sum(k_vecs**2, [2])
    n2 = tf.reduce_sum(n**2, [1])
    
    k_mask_less = tf.cast(tf.less_equal(k2 / angstrom**2, (gcut * 2 * np.pi)**2), dtype = float_type)
    k_mask_zero = tf.cast(tf.not_equal(n2, 0), dtype = float_type)
    
    k_mask = k_mask_less * tf.expand_dims(k_mask_zero, 0)
    
    return k_vecs, k2, k_mask
    
def reciprocal_part(alpha, gcut, charges, positions, rvecs, elements_mask, float_type = tf.float32): 
    gvecs = tf.linalg.inv(rvecs) #  Reciprocal cell matrix
    volume = tf.linalg.det(rvecs) * angstrom**3
    
    k_vecs, k2, k_mask = generate_kvecs(gvecs, rvecs, gcut, float_type = float_type) # [batches, K, 3]

    kr = tf.einsum('ijk,ilk->ijl', k_vecs, positions) # [batches, K, N]
    k2 /= angstrom**2
    
    cos_term = tf.reduce_sum(tf.expand_dims(charges, [1]) * tf.math.cos(kr) * elements_mask, [2])
    sin_term = tf.reduce_sum(tf.expand_dims(charges, [1]) * tf.math.sin(kr) * elements_mask, [2])
    
    rho_k2 = cos_term**2 + sin_term**2
    k_factor = 4 * np.pi / (2 * volume) * tf.exp(- k2 / (4 * alpha**2)) / (k2 + 1. - k_mask) # Avoid dividing by zero
    
    return tf.reduce_sum(rho_k2 * k_factor * k_mask, [1])

