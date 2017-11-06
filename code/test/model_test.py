'''overall model testing'''

__author__ = "innerpeace"


import sys
import logging
sys.path.append('..')
from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from config import Config as cfg
from utils.read_data import read_answers, mask_dataset

import tensorflow.contrib.rnn as rnn
from matchLSTM_cell_test import matchLSTMcell

data_dir = cfg.DATA_DIR
test_file_path = pjoin(root_dir, 'cache', 'test.test_masked.npy')
context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
embed_dim  = 100
num_hidden = cfg.lstm_num_hidden
set_names = cfg.set_names
suffixes = cfg.suffixes

def test_model(num_examples, context=None, question=None, embedding=None, answer=None):
    # TODO: how to feed in data
    context_data = [x[0] for x in context]
    context_masks = [x[1] for x in context]

    question_data = [x[0] for x in question]
    question_masks = [x[1] for x in question]

    answer_start = [x[0] for x in answer]
    answer_end = [x[1] for x in answer]

    tf.reset_default_graph()
    with tf.Session() as sess:
        input_num = num_examples
        # shape [batch_size, context_max_length]
        context = tf.placeholder(tf.int32, (input_num, context_max_len))
        context_m = tf.placeholder(tf.bool, (input_num, context_max_len))
        question = tf.placeholder(tf.int32, (input_num, question_max_len))
        question_m = tf.placeholder(tf.bool, (input_num, question_max_len))
        answer_s = tf.placeholder(tf.int32, (input_num,))
        answer_e = tf.placeholder(tf.int32, (input_num,))
        num_example = tf.placeholder(tf.int32,[],name='batch_size')

        embedding = tf.Variable(embedding, dtype=tf.float32, trainable=False)

        context_embed = tf.nn.embedding_lookup(embedding, context)
        logging.info('shape of context embed {}'.format(context_embed.shape))
        question_embed = tf.nn.embedding_lookup(embedding, question)
        logging.info('shape of question embed {}'.format(question_embed.shape))

        num_hidden = cfg.lstm_num_hidden

        con_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden)
        con_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden)



    pass


def main():
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler('log_test.txt')
    logging.getLogger().addHandler(file_handler)

    answer = read_answers(data_dir)
    dataset = mask_dataset(data_dir, set_names=set_names, suffixes=suffixes)

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    test_model(1,dataset['train_context'][:100], dataset['train_question'][:100], embedding, answer['train_answer'][:100])