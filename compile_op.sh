#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared cell_list/cell_list.cc -o nnfflib/cell_list_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
