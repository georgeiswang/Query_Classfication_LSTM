import sys
import time
import numpy as np
import random
import math
import heapq
import tensorflow as tf
from collections import OrderedDict
from numpy import linalg as LA
from layer import AELayer
import layer

class Autoencoder(object):
    def __init__(self, n_in, n_hidden, sparsity_level = 0.05, sparse_reg = 0.001):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = tf.sigmoid
        self.sparsity_level= np.repeat([0.05], self.n_hidden).astype(np.float32)
        self.sparse_reg = sparse_reg
        self.sen_vec=tf.placeholder(tf.float32, [None, 300])
        rng = np.random.RandomState(random.randint(1, 2**30))
        
        # Adapting learning rate
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})
        
        # Hidden Layer
        self.hiddenLayer = AELayer(rng, n_in, n_hidden, "sigmoid", self.learning_rate, self.batch_grad)
        
        self.params = self.hiddenLayer.params

    def LoadParam(self, weights_file):
        params = np.load(weights_file)
        #print params.shape
        print "data is", params['w_hid'].shape
        a=self.hiddenLayer.W.assign(tf.cast(params['w_hid'],tf.float32))
        b=self.hiddenLayer.b.assign(tf.cast(tf.reshape(params['b_hid'],[1,200]),tf.float32))
        c=self.hiddenLayer.b_prime.assign(tf.cast(tf.reshape(params['b_vis'],[1,300]),tf.float32))

    def kl_divergence(self, p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

    def Forward(self, x_in):
        lin_h = tf.matmul(x_in, self.hiddenLayer.W) + self.hiddenLayer.b
        h = self.activation(lin_h)
        #notice sparsity_level read from numpy can be dtype64, should using astype to float32
        kl_div = self.kl_divergence(self.sparsity_level, h)

        lin_output = tf.matmul(tf.transpose(self.hiddenLayer.W), h) + self.hiddenLayer.b_prime
        return lin_output, kl_div

    def Encode(self, x_in):
        lin_h = tf.matmul(x_in,self.hiddenLayer.W) + self.hiddenLayer.b
        h = self.activation(lin_h)
        return h
   
    def TrainNN(self):
        #sen_vec = T.vector()
        sen_vec= self.sen_vec

        updates = OrderedDict({})
        output1, output2 = self.Forward(sen_vec)
        cost = tf.reduce_sum((sen_vec - output1) ** 2)+ self.sparse_reg * output2

        self.gparams=tf.gradients(cost,self.params)

        for param, gparam in zip(self.params, self.gparams):
            #print param
            updates[self.batch_grad[param]] = self.batch_grad[param] + gparam
        feed_dict = sen_vec
        return cost, feed_dict

    def CompileEncodeFun(self):
        #sen_vec = T.vector()
        sen_vec=self.sen_vec

        updates = OrderedDict({})
        output = self.Encode(sen_vec)
        #f = theano.function([sen_vec], output, updates = updates)
        feed_dict = sen_vec
        #self.encode_fun = {output,feed_dict}

        return output, feed_dict

    def Fit(self, train_data, word_lookup_table, word_id, batch_size = 20, alpha = 0.2, n_epoches = 5, weights_file = '/weights'):
        print 'Compling training function...'
        cost, feed_dict=self.TrainNN()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        M = len(train_data)
        batch_cnt = M/batch_size + 1

        hidden_W = None
        hidden_b = None
        output_W = None
        output_b = None
        best = 0.0

        print 'Start training...'
        sys.stdout.flush()
        for epoch in xrange(n_epoches):
            costs = 0.0
            error = 0.0
            print 'test1'
            start_time = time.time()
            for batch in xrange(0, batch_cnt + 1):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, M)
                print batch
                if start >= M:
                    break
                for index in xrange(start, end):
                    print 'test3'
                    #print 'batch %d index %d' % (batch, index)
                    data = word_lookup_table[train_data[index]].mean(axis = 0)
                    #tmp_cost = train_nn(data)
                    #print train_nn
                    #print data.shape
                    data = np.reshape(data, (300, 1))
                    tmp_cost=sess.run(cost, feed_dict={feed_dict:data})
                    costs += tmp_cost
                for param in self.params:
                    print 'test4'
                   # sess.run(tf.initialize_all_variables())

                    old_param=param
                    oldParam=sess.run(param)
                    #grad=tf.div(self.batch_grad[param],(end-start+1))


                    grad=sess.run(self.batch_grad[param])/(end-start+1)
                    tmp = sess.run(self.learning_rate[param])+grad * grad
                    lr = alpha / (np.sqrt(tmp) + 1.0e-6)
                    new_param = oldParam-lr*grad

                    p=param.assign(new_param)
                    t = self.learning_rate[param].assign(tmp)

                    sess.run(p)
                    sess.run(t)
                    print new_param
                    print sess.run(t)
                for param in self.params:
                    print 'test5'
                    #self.batch_grad[param].set_value(np.zeros_like(self.batch_grad[param].get_value(), dtype=tf.float32))
                    paraGrad=self.batch_grad[param].assign(np.zeros_like(sess.run(self.batch_grad[param]), dtype=np.float32))
                    sess.run(paraGrad)
            end_time = time.time()
            minu = int((end_time - start_time)/60)
            sec = (end_time - start_time) - 60 * minu
            print 'Time: %d min %.2f sec' % (minu, sec)
            cur_cost = costs/M
            print 'Traning at epoch %d, cost = %f' % (epoch + 1, cur_cost)
            sys.stdout.flush()

        w_hid = sess.run(self.hiddenLayer.W)
        b_hid = sess.run(self.hiddenLayer.b)
        b_vis = sess.run(self.hiddenLayer.b_prime)

        w_hid, b_hid, b_vis = self.params

        np.savez(weights_file, w_hid = w_hid, b_hid = b_hid, b_vis = b_vis)
