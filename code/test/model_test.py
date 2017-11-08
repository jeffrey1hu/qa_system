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
        # logging.info('hidden stats size of context lstm is {}'.format(con_lstm_fw_cell.state_size))

        con_outputs, con_outputs_states = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,con_lstm_bw_cell,
                                                        context_embed,
                                                        sequence_length=sequence_length(context_m),
                                                        dtype=tf.float32, scope='con_lstm')

        logging.info('the shape of context bilstm outputs is a {} length list with {} tensor elements'
                     .format(len(con_outputs), con_outputs[0].get_shape().as_list()))

        H_context = tf.concat(con_outputs, axis=2)
        logging.info('the shape of h_context is {}'.format(H_context.get_shape().as_list()))

        assert (num_examples, context_max_len, 2 * num_hidden) == H_context.shape, \
            'the shape of H_context should be {} but it is {}'\
                .format((num_examples, context_max_len, 2 * num_hidden), H_context.shape)

        question_lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        question_lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        question_outputs, question_outputs_states = tf.nn.bidirectional_dynamic_rnn(question_lstm_fw_cell,
                                                                     question_lstm_bw_cell,
                                                                     question_embed,
                                                                     sequence_length=sequence_length(question_m),
                                                                     dtype=tf.float32, scope="question_lstm")

        H_question = tf.concat(question_outputs, axis=2)
        logging.info('the shape of h_question is {}'.format(H_question.get_shape().as_list()))

        assert (num_examples, question_max_len, 2 * num_hidden) == H_question.shape, \
            'the shape of H_context should be {} but it is {}'\
                .format((num_examples, question_max_len, 2 * num_hidden), H_question.shape)

        match_lstm_fw_cell = matchLSTMcell(2 * num_hidden, 2 * num_hidden, H_question, question_m)
        match_lstm_bw_cell = matchLSTMcell(2 * num_hidden, 2 * num_hidden, H_question, question_m)

        match_outputs, _ = tf.nn.bidirectional_dynamic_rnn(match_lstm_fw_cell,
                                                     match_lstm_bw_cell,
                                                     H_context,
                                                     sequence_length=sequence_length(context_m),
                                                     dtype=tf.float32, scope='match_lstm')
        H_r = tf.concat(match_outputs, axis=2)

        logging.info('the shape of h_r is {}'.format(H_r.get_shape().as_list()))

        assert (num_examples, context_max_len, 2 * num_hidden) == H_context.shape, \
            'the shape of H_r should be {} but it is {}'.format((num_examples, context_max_len, 2 * num_hidden),
                                                                                          H_context.shape)
        # Decoder session
        H_r_shape = H_r.get_shape().as_list()
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decoder"):
            W_r = tf.get_variable("V_r", shape=[num_hidden * 4, num_hidden * 2], dtype=tf.float32, initializer=initializer)

            W_f = tf.get_variable("W_f", shape=[num_hidden * 2, 1], dtype=tf.float32, initializer=initializer)

            W_h = tf.get_variable("W_h", shape=[num_hidden * 4, num_hidden * 2], dtype=tf.float32, initializer=initializer)

            B_r = tf.get_variable("B_r", shape=[num_hidden * 2], dtype=tf.float32)
            B_f = tf.get_variable("B_f", shape=[], dtype=tf.float32)

            W_r_e = tf.tile(tf.expand_dims(W_r, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_f_e = tf.tile(tf.expand_dims(W_f, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_h_e = tf.tile(tf.expand_dims(W_h, axis=0), multiples=[H_r_shape[0], 1, 1])

            # f1 -> (b, q, 2n)
            f1 = tf.nn.tanh(tf.matmul(H_r, W_r_e) + B_r)
            # s_score -> (b, q, 1)
            s_score = tf.matmul(f1, W_f_e) + B_f
            # s_score -> (b, q)
            s_score = tf.squeeze(s_score, axis=2)
            logging.info('shape of s_score is {}'.format(s_score.shape))

            # the prob distribution of start index
            s_prob = tf.nn.softmax(s_score)
            s_prob = s_prob * context_m
            # Hr_attend -> (batch_size, 4n)
            Hr_attend = tf.reduce_sum(H_r * tf.expand_dims(s_prob, axis=2), axis=1)

            f2 = tf.nn.tanh(tf.matmul(H_r, W_r_e)
                            + tf.matmul(tf.tile(tf.expand_dims(Hr_attend, axis=1), multiples=[1, H_r_shape[1], 1]), W_h_e)
                            + B_r)

            e_score = tf.matmul(f2, W_f_e) + B_f
            e_score = tf.squeeze(e_score, axis=2)
            logging.info('shape of e_score is {}'.format(e_score.shape))

            # e_prob = tf.nn.softmax(e_score)
            # e_prob = tf.multiply(e_prob, context_m)

            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_s, logits=s_score)
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_e, logits=e_score)

            final_loss = tf.reduce_mean(loss_e + loss_s)

            train_op = tf.train.AdadeltaOptimizer().minimize(final_loss)
            outputs = [H_context, H_question, H_r, s_score, e_score, s_prob, answer_s, answer_e,loss_s,loss_e, final_loss]
            sess.run(tf.global_variables_initializer())
            out = sess.run(outputs, feed_dict={context:context_data, context_m:context_masks,
                                                                  question: question_data, question_m:question_masks,
                                                                  answer_s:answer_start, answer_e:answer_end,
                                                                  num_example:num_examples})
            print('test success.')
            logging.info('loss is : {}'.format(out[-1]))
            logging.info('H_context: {}'.format(out[0]))
            logging.info('H_question: {}'.format(out[1]))
            logging.info('H_r : {}'.format(out[2]))
            logging.info('s_score: {}'.format(out[3]))
            logging.info('e_score: {}'.format(out[4]))
            logging.info('prob_s : {}'.format(out[5]))
            logging.info('answer prob_s : {}'.format(out[5][0][out[6]]))
            logging.info('answers : {}, {}'.format(out[6], out[7]))
            logging.info('loss: {}, {}'.format(out[8], out[9]))

    pass


def sequence_length(sequence_mask):
    """
    Args:
        sequence_mask: Bool tensor with shape -> [batch_size, q]

    Returns:
        tf.int32, [batch_size]

    """
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)


def main():
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler('log_test.txt')
    logging.getLogger().addHandler(file_handler)

    answer = read_answers(data_dir)
    dataset = mask_dataset(data_dir, set_names=set_names, suffixes=suffixes)

    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)['glove']

    test_model(1,dataset['train_context'][:100], dataset['train_question'][:100], embedding, answer['train_answer'][:100])


if __name__ == '__main__':
    main()