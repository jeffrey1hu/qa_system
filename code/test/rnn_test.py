'''rnn test'''

__author__ = 'innerpeace'

import sys
import logging
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from config import Config as cfg
from utils.read_data import mask_dataset
import tensorflow.contrib.rnn as rnn

data_dir = cfg.DATA_DIR
set_names = cfg.set_names
suffixes = cfg.suffixes

def rnn_test():

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    dataset = mask_dataset(data_dir, set_names=set_names, suffixes=suffixes)
    test_data = dataset['train_context'][:2]
    inputs = [x[0] for x in test_data]
    masks = [x[1] for x in test_data]

    inputs = np.array(inputs)
    print('shape of inputs {}'.format(inputs.shape))
    masks = np.array(masks)
    print('shape of masks {}'.format(masks.shape))

    with tf.Graph().as_default():
        # embedding_tf = tf.Variable(embedding)
        x = tf.placeholder(tf.int32, (None, 400))
        x_m = tf.placeholder(tf.bool, (None, 400))
        l_x = tf.placeholder(tf.int32, (None,))
        print(x)
        print(x_m)
        print(l_x)

        embed = tf.nn.embedding_lookup(embedding, x)
        # x_in = tf.boolean_mask(embed, x_m)
        print('shape of embed {}'.format(embed.shape))
        # print('shape of x_in {}'.format(x_in.shape))

        num_hidden = 5
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
                                                              embed,sequence_length=sequence_length(x_m),dtype=tf.float64)
        outputs = tf.concat(outputs, axis=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outp, outps = sess.run([ outputs, outputs_states], feed_dict={x:inputs,
                                                                         x_m:masks})
            # print('shape of input embeddings is : {}'.format(xin.shape))
            print("shape of output is :{}".format(outp.shape))
            assert outp.shape == (2, 400, 2 * num_hidden), 'the shape of outp should be {} but it is {}'\
                .format((2, 400, 2 * num_hidden), outp.shape)
            print(outp)

def sequence_length(sequence_mask):
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rnn_test()
