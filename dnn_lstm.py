import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import random
import math
import heapq
from collections import OrderedDict
from layer import Layer

#rewritte by tensorflow
class DNN(object):
    def __init__(self, n_hidden, n_out, reg_exp_size, ae_size, id_to_reg_exp,
                 id_to_word, word_lookup_table, auto_encoder, L2_reg=0.0001):
       # sess = tf.Session()

        self.n_hidden = n_hidden
        self.n_out = n_out
        self.L2_reg = L2_reg
        self.activation = tf.tanh #modification 1
        self.auto_encoder = auto_encoder
        self.word_lookup_table = word_lookup_table
        self.id_to_word = id_to_word
        self.id_to_reg_exp = id_to_reg_exp
        rng = np.random.RandomState(random.randint(1, 2 ** 30))

        # Adapting learning rate
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})

        # word dict size and ner dict size and reg_exp_dict size
        self.ae_size = ae_size
        self.reg_V = reg_exp_size

        self.x_in=tf.placeholder(tf.float32, shape=(None, 20, 200))#memory size is 5
        self.reg_x=tf.placeholder(tf.int32, shape=(None,))
        self.y=tf.placeholder(tf.int32)
        self.i=0

        # Skip Layer for encoder
        # The detailed tensorflow structure is used in Layer method

        self.skip_layer_ae = Layer(rng, ae_size, n_out, "tanh", self.learning_rate, self.batch_grad)
        # Skip Layer for reg,
        self.skip_layer_re = Layer(rng, self.reg_V, n_out, "tanh", self.learning_rate, self.batch_grad)
        # Hidden Layer, ae_size=n_hidden=200
        self.hiddenLayer = Layer(rng, ae_size, n_hidden, "tanh", self.learning_rate, self.batch_grad)
        # Output Layer
        self.outputLayer = Layer(rng, n_hidden, n_out, "tanh", self.learning_rate, self.batch_grad)

        # Lookup table for reg
        """
        reg_lookup_table_value = rng.uniform(low=-0.01, high=0.01, size=(self.reg_V, n_hidden))
        reg_lookup_table_value = np.asarray(reg_lookup_table_value, dtype=theano.config.floatX)
        self.reg_lookup_table = theano.shared(value=reg_lookup_table_value, name='rlt', borrow=True)
        self.learning_rate[self.reg_lookup_table] = theano.shared(value=np.ones(reg_lookup_table_value.shape,
                                                                                dtype=theano.config.floatX),
                                                                  borrow=True)
        self.batch_grad[self.reg_lookup_table] = theano.shared(value=np.zeros(reg_lookup_table_value.shape,
                                                                              dtype=theano.config.floatX), borrow=True)
        """
        reg_lookup_table_value = rng.uniform(low=-0.01, high=0.01, size=(self.reg_V, n_hidden))
        self.reg_lookup_table = tf.Variable(np.asarray(reg_lookup_table_value), dtype=tf.float64, name='rlt')
        self.learning_rate[self.reg_lookup_table]=tf.Variable(np.ones(reg_lookup_table_value.shape),dtype=tf.float64, name='learnrate')

        print (reg_lookup_table_value.shape)
        self.batch_grad[self.reg_lookup_table]=tf.Variable(np.zeros(reg_lookup_table_value.shape),dtype=tf.float64,name='batchgrad')
        self.params = self.hiddenLayer.params + self.outputLayer.params + self.skip_layer_ae.params + self.skip_layer_re.params + [
            self.reg_lookup_table]

        #sess.run(tf.initialize_all_variables())

    def LoadParam(self, weights_file,sess):
        params = np.load(weights_file)
        #notice that W here is a tensor variable declared in the layer file
        sess.run(self.hiddenLayer.W.assign(params['hidden_W']))
        sess.run(self.hiddenLayer.b.assign(params['hidden_b']))
        sess.run(self.outputLayer.W.assign(params['output_W']))
        sess.run(self.outputLayer.b.assign(params['output_b']))
        sess.run(self.skip_layer_ae.W.assign(params['skip_ae_W']))
        sess.run(self.skip_layer_ae.b.assign(params['skip_ae_b']))
        sess.run(self.skip_layer_re.W.assign(params['skip_re_W']))
        sess.run(self.skip_layer_re.b.assign(params['skip_re_b']))
        sess.run(self.reg_lookup_table.assign(params['reg_lookup_table']))

    def SaveMatrix(self, f, matrix):
        r = matrix.shape[0]
        c = matrix.shape[1]

        f.write('%d %d\n' % (r, c))
        for line in matrix:
            ret = [str(val) for val in line]
            ret = ' '.join(ret)
            f.write("%s\n" % ret)

    def SaveVector(self, f, vector):
        r = vector.shape[0]

        f.write('%d\n' % r)
        ret = [str(val) for val in vector]
        ret = ' '.join(ret)
        f.write("%s\n" % ret)

    def SaveLookupTable(self, f, lt, id_to_name):
        r = lt.shape[0]
        c = lt.shape[1]

        f.write('%d %d\n' % (r, c))
        for i in xrange(len(lt)):
            ret = [str(val) for val in lt[i]]
            ret = ' '.join(ret)
            f.write("%s %s\n" % (id_to_name[i].encode('utf-8'), ret))

    def SaveParam(self, weights_file_dir, sess):
        word_lookup_table_file = weights_file_dir + 'word_lookup_table.txt'
        f = open(word_lookup_table_file, 'w')
        self.SaveLookupTable(f, self.word_lookup_table, self.id_to_word)
        f.close()
        reg_lookup_table_file = weights_file_dir + 'reg_lookup_table.txt'
        f = open(reg_lookup_table_file, 'w')
        self.SaveLookupTable(f, sess.run(self.reg_lookup_table), self.id_to_reg_exp)
        f.close()
        dnn_weights_file = weights_file_dir + 'dnn_layer_weights.txt'
        f = open(dnn_weights_file, 'w')
        # encode layer
        self.SaveMatrix(f, sess.run(self.auto_encoder.hiddenLayer.W))
        self.SaveVector(f, sess.run(self.auto_encoder.hiddenLayer.b))
        # hidden layer
        self.SaveMatrix(f, sess.run(self.hiddenLayer.W))
        self.SaveVector(f, sess.run(self.hiddenLayer.b))
        # output layer
        self.SaveMatrix(f, sess.run(self.outputLayer.W))
        self.SaveVector(f, sess.run(self.outputLayer.b))
        # skip layer for ae
        self.SaveMatrix(f, sess.run(self.skip_layer_ae.W))
        self.SaveVector(f, sess.run(self.skip_layer_ae.b))
        # skip layer for re
        self.SaveMatrix(f, sess.run(self.skip_layer_re.W))
        self.SaveVector(f, sess.run(self.skip_layer_re.b))
        f.close()

    def Forward(self, sess):
        lstm= tf.nn.rnn_cell.BasicLSTMCell(200, forget_bias=1.0)#LSTM size
        #lstm=tf.nn.rnn_cell.GRUCell(10)
        state=tf.zeros([1,200])# batch size, state_num=2*step_size
        num_steps=20# we don't need time step actually, the length of sentence is time-step
        x_in_batch = tf.transpose(self.x_in, [1, 0, 2])#change to 20*1*200
        x_in = tf.reshape(x_in_batch, [-1, 200])#change to 20*200
        x_in = tf.split(0, 20, x_in)#this will return a list, i.e. 20 sequences of 1*200

        if self.i == 0:
            with tf.variable_scope('output'):
                output_lstm, state=rnn.rnn(lstm, x_in, dtype=tf.float32)
                #output_lstm, state= lstm(x_in,state)#200*1
        else:
            with tf.variable_scope('output', reuse=True):
                output_lstm, state = rnn.rnn(lstm, x_in, dtype=tf.float32)
                #output_lstm, state= lstm(x_in,state)
        self.i+=1

        output_lstm=output_lstm[-1]# get the last element of a list

        lin_h=tf.matmul(output_lstm,self.hiddenLayer.W)+self.hiddenLayer.b
        #x_in=1*200, W=200*200

        reg_h = tf.reduce_sum(tf.gather(self.reg_lookup_table, self.reg_x), 0)#Num*200
        print "reg_h is"
        print reg_h
        h = self.activation(lin_h + tf.cast(reg_h,tf.float32))#1*200

        lin_output_pre = tf.matmul(h, self.outputLayer.W) + self.outputLayer.b
        lin_output = tf.nn.dropout(lin_output_pre, keep_prob=0.6)

        #h=1*200, outputLayer.W=200*63, lin_outupt=1*63
        #re.W:19156*63
        reg_output = tf.reduce_sum(tf.gather(self.skip_layer_re.W, self.reg_x), 0) + self.skip_layer_re.b
        print reg_output

        #x_in=1*200. ae.W=200*63
        ae_output = tf.matmul(x_in[-1], self.skip_layer_ae.W) + self.skip_layer_ae.b#use the last element as skip layer input
        ae_output = tf.nn.dropout(ae_output, keep_prob=0.5)

        output = tf.nn.softmax(lin_output + ae_output + reg_output)#XXX*63

        return output

    def TrainNN(self,sess):
        gradient_step=0.1
        output = self.Forward(sess)

        cost=-tf.gather(tf.log(tf.gather(tf.transpose(output),self.y)),0)
        train_op = tf.train.GradientDescentOptimizer(gradient_step).minimize(cost)
        train_op = tf.train.AdagradOptimizer(gradient_step).minimize(cost)

        return train_op, output, cost

    def GetResult(self,sess):

        output = self.Forward(sess)

        return tf.gather(output,0)

    def GetTopK(self, output, k):
        result = [t[0] for t in heapq.nlargest(k, enumerate(output), lambda t: t[1])]
        return result


    def Fit(self, train_data, train_ans, dev_data, dev_ans, sess, batch_size=10, alpha=0.5, n_epoches=25,
            weights_file_dir='mlp_output/'):
        print 'Compling training function...'

        train_fetch, output_fetch, cost_fetch= self.TrainNN(sess)
        get_output = self.GetResult(sess)
        sess.run(tf.initialize_all_variables())

        M = len(train_data)
        N = len(dev_data)
        batch_cnt = M / batch_size + 1

        hidden_W = None
        hidden_b = None
        output_W = None
        output_b = None
        skip_ae_W = None
        skip_ae_b = None
        skip_re_W = None
        skip_re_b = None
        reg_lookup_table = None
        best = 0.0
        memorySize=5 #LSTM memory size
##########################
##########################

        print 'Start training...'
        sys.stdout.flush()
        '''
        # combining data into 1000
        for index in xrange(0, M-memorySize):
            trainDataLSTM = np.transpose(np.asarray(train_data[index][0]))
            regLSTM = np.transpose(np.asarray(train_data[index][1]))  # has to transpose to use concatenate
            for idx in xrange(index + 1, index + memorySize):
                if np.asarray(train_data[idx][0]).shape == (1, 200):
                    trainDataLSTM = np.concatenate((trainDataLSTM, np.asarray(np.transpose(train_data[idx][0]))))
                else:
                    trainDataLSTM = np.concatenate((trainDataLSTM, np.asarray(train_data[idx][0])))

                regLSTM = np.concatenate((regLSTM, np.asarray(np.transpose(train_data[idx][1]))))
            trainDataInput = np.concatenate(trainDataInput,np.transpose(trainDataLSTM))
            regLSTMInput = np.concatenate(regLSTMInput,np.transpose(regLSTM))
        '''


        for epoch in xrange(n_epoches):
            costs = 0.0
            error = 0.0
            rightTrain=M
            totalTrain=M
            rightDev=N
            totalDev=N
            start_time = time.time()
            for batch in xrange(0, batch_cnt + 1):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, M)
                if start >= M:
                    break
                for index in xrange(start, end):
                    print 'Epoch'+repr(epoch)+'Batch' +repr(batch)+','+'index is'+repr(index)
                    if np.asmatrix(train_data[index][0]).shape[1] == 200:
                        [tmp_cost, train_op] = sess.run([cost_fetch, train_fetch], feed_dict={self.x_in:np.expand_dims(np.asmatrix(train_data[index][0]),axis=0), self.reg_x:train_data[index][1], self.y:(train_ans[index])})
                    else:
                        [tmp_cost, train_op] = sess.run([cost_fetch, train_fetch], feed_dict={self.x_in:np.expand_dims(np.transpose(np.asmatrix(train_data[index][0])),axis=0),self.reg_x: train_data[index][1],self.y: (train_ans[index])})
                    costs += tmp_cost
                    print 'temp cost is'
                    print tmp_cost

                print 'Training Cost is'+ repr(costs)

            end_time = time.time()
            minu = int((end_time - start_time) / 60)
            sec = (end_time - start_time) - 60 * minu
            print 'Time: %d min %.2f sec' % (minu, sec)
            cur_cost = costs / M
            #print 'Traning at epoch=%d, cost = %f' % (epoch + 1, cur_cost)

            for index in xrange(0, M):
                if np.asmatrix(train_data[index][0]).shape[1]==200:
                    tempRight = ((train_ans[index]) == np.argmax(sess.run(get_output,feed_dict={self.x_in:np.expand_dims(np.asmatrix(train_data[index][0]),axis=0), self.reg_x:train_data[index][1]})))
                else:
                    tempRight = ((train_ans[index]) == np.argmax(sess.run(get_output,feed_dict={self.x_in:np.expand_dims(np.transpose(np.asmatrix(train_data[index][0])),axis=0), self.reg_x:train_data[index][1]})))+1
                rightTrain+=tempRight
            totalTrain+=M
            pre = (1.0 * rightTrain) / totalTrain * 100
            print 'Train pre: %f' % pre

            for index in xrange(0, N):
                if np.asmatrix(dev_data[index][0]).shape[1] == 200:
                    tempRight = ((dev_ans[index]) == np.argmax(sess.run(get_output,feed_dict={self.x_in:np.expand_dims(np.asmatrix(dev_data[index][0]),axis=0), self.reg_x:(dev_data[index][1])})))
                else:
                    tempRight = ((dev_ans[index]) == np.argmax(sess.run(get_output, feed_dict={self.x_in: np.expand_dims(np.transpose(np.asmatrix(dev_data[index][0])),axis=0), self.reg_x:(dev_data[index][1])})))+1
                rightDev+=tempRight
            totalDev+=N
            pre = (1.0 * rightDev) / totalDev* 100
            print 'Dev pre: %f' % pre
            sys.stdout.flush()
            if pre > best:
                best = pre
                hidden_W = sess.run(self.hiddenLayer.W)
                hidden_b = sess.run(self.hiddenLayer.b)
                output_W = sess.run(self.outputLayer.W)
                output_b = sess.run(self.outputLayer.b)
                skip_ae_W = sess.run(self.skip_layer_ae.W)
                skip_ae_b = sess.run(self.skip_layer_ae.b)
                skip_re_W = sess.run(self.skip_layer_re.W)
                skip_re_b = sess.run(self.skip_layer_re.b)
                reg_lookup_table = sess.run(self.reg_lookup_table)
                # np.savez(weights_file, hidden_W = hidden_W, hidden_b = hidden_b, output_W = output_W, output_b = output_b,
                # skip_ae_W = skip_ae_W, skip_ae_b = skip_ae_b, skip_re_W = skip_re_W, skip_re_b = skip_re_b, reg_lookup_table = reg_lookup_table)
        sess.run(self.hiddenLayer.W.assign(hidden_W))
        sess.run(self.hiddenLayer.b.assign(hidden_b))
        sess.run(self.outputLayer.W.assign(output_W))
        sess.run(self.outputLayer.b.assign(output_b))
        sess.run(self.skip_layer_ae.W.assign(skip_ae_W))
        sess.run(self.skip_layer_ae.b.assign(skip_ae_b))
        sess.run(self.skip_layer_re.W.assign(skip_re_W))
        sess.run(self.skip_layer_re.b.assign(skip_re_b))
        sess.run(self.reg_lookup_table.assign(reg_lookup_table))
        #print 'Saving results...'
        #self.SaveParam(weights_file_dir,sess)
