# -*- coding: utf-8 -*-
#
# Satelight.py
#
# Created by Kosuke FUKUMORI on 2021/02/19
#

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm

class SimpleSelfAttention(keras.Model):
    
    def __init__(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.q_dense_layer = keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = keras.layers.Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = keras.layers.Dense(depth, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')

    def call(self, input: tf.Tensor) -> tf.Tensor:
        memory = input
        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory) 
        v = self.v_dense_layer(memory)
        logit = tf.matmul(q, k, transpose_b=True)

        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)


def Satelight(segment_length: int, 
              n_channels: int,
              fact_conv_size: int = 250,
              n_fact_conv: int = 16,
              D: int = 2,
              dropout_rate: float = 0.2,
              n_hop: int = 3,
              ):
    
    input_ = keras.layers.Input(shape=(n_channels, segment_length, 1))
    
    '''
    Embedding block
    Input shape is (n_channels, segment_length, 1)
    Output shape shape will be (segment_length//4, n_fact_conv*D)
    '''
    block1 = keras.layers.Conv2D(n_fact_conv, (1, fact_conv_size),
                                 padding='same', use_bias=False,
                                 )(input_)
    block1 = keras.layers.DepthwiseConv2D((n_channels, 1), use_bias=False,
                                          depth_multiplier=D, depthwise_constraint=max_norm(1.),
                                          )(block1)
    block1 = keras.layers.BatchNormalization()(block1)
    block1 = keras.layers.Activation('relu')(block1)
    block1 = keras.layers.Dropout(dropout_rate)(block1)
    block1 = keras.layers.AveragePooling2D((1, 4))(block1)
    block1 = keras.layers.Reshape((-1, n_fact_conv*D))(block1)
    
    '''
    SA blocks
    Input shape is (segment_length//4, n_fact_conv*D)
    Output shape shape will be (segment_length//4//(4**n_hop), n_fact_conv*D*(n_hop+1))
    '''
    block2 = block1
    for hop in range(n_hop):
        skip = block2
        block2 = SimpleSelfAttention(n_fact_conv*D*(hop+1))(block2)
        block2 = keras.layers.BatchNormalization(axis=-1)(block2)
        block2 = keras.layers.Dropout(dropout_rate)(block2)
        block2 = keras.layers.Add()([skip, block2])
        
        skip = block2
        zeros = tf.keras.backend.zeros_like(skip)
        zeros = tf.split(zeros, num_or_size_splits=hop+1, axis=2)[0]
        skip = tf.keras.backend.concatenate([skip, zeros], axis=-1)
        
        block2 = keras.layers.Dense(n_fact_conv*D*(hop+2), use_bias=True)(block2)
        block2 = keras.layers.BatchNormalization()(block2)
        block2 = keras.layers.Activation('relu')(block2)
        block2 = keras.layers.Dropout(dropout_rate)(block2)
        block2 = keras.layers.Add()([skip, block2])
        
        block2 = keras.layers.AveragePooling1D(4)(block2)
    
    
    flatten = keras.layers.Flatten(name='flatten')(block2)
    dense = keras.layers.Dense(1, name='output_dense')(flatten)
    softmax = keras.layers.Activation('sigmoid', name='softmax')(dense)

    return keras.Model(inputs=input_, outputs=softmax)

def transposer(X: np.ndarray):
    return X.transpose([0, 2, 1])[:,:,:,None]