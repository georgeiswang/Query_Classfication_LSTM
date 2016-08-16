import numpy as np
import tensorflow as tf
# modified by tensorflow, Done by Yu Wang --2016/7/1
class Layer(object):
    def __init__(self, rng, n_in, n_out, activation_type, learning_rate, batch_grad):

        #n_in=300, n_out=200 for sparse
        #n_in=200, n_out=? for DNN
        W=tf.Variable(tf.random_uniform([n_in, n_out],minval=-tf.sqrt(tf.cast(6./(n_in+n_out),tf.float32)), maxval=tf.sqrt(tf.cast((6./(n_in+n_out)),tf.float32)),dtype=tf.float32), name= 'W')
        b=tf.Variable(tf.zeros((n_out,1), dtype=tf.float32), name='b')
        b = tf.Variable(tf.zeros((1,n_out), dtype=tf.float32), name='b1')

        #why activation_type will change the sigmoid weights
        if activation_type == 'sigmoid':
            W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-4*tf.sqrt(tf.cast(6. / (n_in + n_out), tf.float32)),
                                              maxval=4*tf.sqrt(tf.cast((6. / (n_in + n_out)), tf.float32)),
                                              dtype=tf.float32), name='W')

        self.W = W
        self.b = b
        #self.b1 = b1

        learning_rate[self.W]=tf.Variable(tf.ones([n_in,n_out],dtype=tf.float32))
        learning_rate[self.b]=tf.Variable(tf.ones((n_out,1),dtype=tf.float32))
        batch_grad[self.W]=tf.Variable(tf.zeros([n_in, n_out], dtype = tf.float32))
        batch_grad[self.b] =tf.Variable(tf.zeros((n_out,1), dtype=tf.float32))

        self.params = [self.W, self.b]


class AELayer(Layer):
    def __init__(self, rng, n_in, n_out, activation_type, learning_rate, batch_grad):
        Layer.__init__(self, rng, n_in, n_out, activation_type, learning_rate, batch_grad)

        b_prime=tf.Variable(tf.zeros((1,n_in),dtype=tf.float32))

        self.b_prime = b_prime

        learning_rate[self.b_prime]=tf.Variable(tf.ones((n_in,1),dtype=tf.float32))
        batch_grad[self.b_prime]=tf.Variable(tf.ones((n_in,1), dtype=tf.float32))

        self.params = [self.W, self.b, self.b_prime]
