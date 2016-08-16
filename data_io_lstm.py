import numpy as np
import tensorflow as tf
import random
def ReadWordLookupTable(lookup_table_file):
    #Find the word in word2vec 300 table
    lt = []
    word_id = {}
    id_to_word = []
    V = 0
    f = open(lookup_table_file, 'r')
    f.readline() # omit the header
    for line in f:
        line = line.strip().decode('utf-8')
        line = line.split()
        tmp = [float(t) for t in line[1:len(line)]]
        lt.append(tmp)
        word_id[line[0]] = V
        id_to_word.append(line[0])
        V += 1
    f.close()
    return (lt, word_id, id_to_word)

def ReadLable(label_id_file):
    label_id = {}
    id_to_label = []
    V = 0
    f = open(label_id_file, 'r')
    for line in f:
        line = line.strip().decode('utf-8')
        label_id[line] = V
        id_to_label.append(line)
        V += 1
    return (label_id, id_to_label)

def ReadFeature(feature_file, label_id, word_id, reg_exp_dict, id_to_reg_exp, ae_hidden_layer_size, word_lookup_table, auto_encoder,sess, mode):
    data = []
    ans = []
    f = open(feature_file,'r')
    count=0;
    '''
    for line in f:# reading in the word segment in batches
        line = line.strip().decode('utf-8')
        line = line.split('\t')
        unigram = []
        senPack=[]
        for word in line[1].split(' '):
            if word not in word_id:
                continue
            unigram.append(word_id[word])

        if len(unigram) == 0:
            rep = np.zeros(ae_hidden_layer_size, dtype=np.float32)
        else:
            sen = word_lookup_table[unigram].mean(axis=0)
            sen = np.asmatrix(sen)
        senPack.append(sen)
        print len(senPack)
    '''
    output, feed_dict = auto_encoder.CompileEncodeFun()
    sess.run(tf.initialize_all_variables())

    for line in f:
        count=count+1
        line = line.strip().decode('utf-8')
        line = line.split('\t')
        tag = line[0]
        # unigram feature
        unigram = []
        senPack = []
        for word in line[1].split(' '):
            if word not in word_id:
                continue
            unigram.append(word_id[word])

        if len(unigram) == 0:
            rep = np.zeros(ae_hidden_layer_size, dtype=np.float32)
        else:
            sen = word_lookup_table[unigram]#using data directly
            sen = np.asmatrix(sen)
            if sen.shape[0]>=20:
                sen=sen[0:20,:]
            else:
                padded=random.random()*np.ones((20,sen.shape[1]))
                padded[0:sen.shape[0], 0:sen.shape[1]] = sen
                sen=padded

            # reg_exp feature
        reg_exp = []
        for reg in line[2].split(' '):
            if reg not in reg_exp_dict:
                reg_exp_dict[reg] = len(reg_exp_dict)# reg_exp_dict store the length of each expression
                id_to_reg_exp.append(reg)
                print 'the reg expression is'+ repr(reg)
                #print repr(id_to_reg_exp)
            reg_exp.append(reg_exp_dict[reg])


        rep = sess.run(output, feed_dict={feed_dict: sen})
        rep = np.asarray(rep, dtype=np.float32)

        data.append((rep, reg_exp))
        ans.append(label_id[tag]) # give a number of label_id in the label list here
        print 'Round No. is' + repr(count)

        if (mode==1) and (count>100000): #mode 1 is for training
            break
        if (mode==2) and (count>20000):# mode 2 is for testing
            break
    f.close()
    return (data, ans)# ans is the label of corresponding category

def ReadUnlabelData(data_file, word_id):
    f = open(data_file,'r')
    train_data = []
    for line in f:
        line = line.strip().decode('utf-8')
        line = line.split(' ')
        sen = []
        for word in line:
            if word in word_id:
                sen.append(word_id[word])
        if len(sen) == 0:
            continue
        train_data.append(sen)
    f.close()
    return train_data


