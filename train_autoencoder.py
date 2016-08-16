import sys
import numpy as np
#import theano
import data_io
import tensorflow as tf
from sparse_autoencoder import Autoencoder

def TrainAutoencoder(conf):
    train_data_file = conf.GetKey('query_log_file_path')
    word_lookup_table_file = conf.GetKey('word_lookup_table_file_path')
    sparsity_level = float(conf.GetKey('sparsity_level'))
    sparse_reg = float(conf.GetKey('sparse_reg'))
    n_hidden = int(conf.GetKey('ae_hidden_layer_size'))
    batch_size = int(conf.GetKey('ae_batch_size'))
    learning_rate = float(conf.GetKey('ae_learning_rate'))
    n_epoches = int(conf.GetKey('ae_n_epoches'))
    weights_file = conf.GetKey('ae_param_file_path')

    print 'Reading lookup table...'
    (word_lookup_table, word_id, id_to_word) = data_io.ReadWordLookupTable(word_lookup_table_file)
    print 'Reading training data...'
    word_lookup_table = np.asarray(word_lookup_table, dtype = np.float32)
    train_data = data_io.ReadUnlabelData(train_data_file, word_id)
    n_in = len(word_lookup_table[0])
    dnn = Autoencoder(n_in, n_hidden, sparsity_level, sparse_reg)
    dnn.Fit(train_data, word_lookup_table, word_id, batch_size, learning_rate, n_epoches, weights_file)
