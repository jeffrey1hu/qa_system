from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from utils.matchLSTM_cell import matchLSTMcell
from config import Config as cfg

logging.basicConfig(level=logging.INFO)

context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
n_hidden = cfg.n_hidden
dtype = cfg.dtype
keep_prob = cfg.keep_prob
learning_rate = cfg.start_lr
max_grad_norm = cfg.max_grad_norm


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def sequence_length(sequence_mask):
    """
    Args:
        sequence_mask: Bool tensor with shape -> [batch_size, q]

    Returns:
        tf.int32, [batch_size]

    """
    return tf.reduce_sum(tf.cast(sequence_mask, tf.int32), axis=1)


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, context, context_m, question, question_m, embedding):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # context shape -> (None, P)
        # context embed -> (None, P, n)
        context_embed = tf.nn.embedding_lookup(embedding, context)

        question_embed = tf.nn.embedding_lookup(embedding, question)

        with tf.variable_scope('context_lstm'):
            con_lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            con_lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            # shape of outputs -> [output_fw, output_bw] -> output_fw -> [batch_size, P, n]
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(con_lstm_fw_cell,
                                                         con_lstm_bw_cell,
                                                         context_embed,
                                                         sequence_length=sequence_length(context_m),
                                                         dtype=dtype)

        # need H_context as dim (Batch_size, hidden_size, P)
        # dimension of outputs
        with tf.variable_scope('H_context'):
            # H_context -> (batch_size, P, 2n)
            H_context = tf.concat(outputs, axis=2)
            H_context = tf.nn.dropout(H_context, keep_prob=keep_prob)

        with tf.variable_scope('question_lstm'):
            question_lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            question_lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            # shape of outputs -> [output_fw, output_bw] -> output_fw -> [batch_size, P, n]
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(question_lstm_fw_cell,
                                                         question_lstm_bw_cell,
                                                         question_embed,
                                                         sequence_length=sequence_length(question_m),
                                                         dtype=dtype)

        with tf.variable_scope('H_question'):
            # H_question -> (batch_size, Q, 2n)
            H_question = tf.concat(outputs, axis=2)
            H_question = tf.nn.dropout(H_question, keep_prob=keep_prob)

        with tf.variable_scope('H_match_lstm'):
            match_lstm_fw_cell = matchLSTMcell(2 * n_hidden, self.size, H_question, question_m)
            match_lstm_bw_cell = matchLSTMcell(2 * n_hidden, self.size, H_question, question_m)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(match_lstm_fw_cell,
                                                         match_lstm_bw_cell,
                                                         H_context,
                                                         sequence_length=sequence_length(context_m),
                                                         dtype=dtype)

        # H_match -> (batch_size, Q, 2n)
        with tf.variable_scope('H_match'):
            H_r = tf.concat(outputs, axis=2)
            H_r = tf.nn.dropout(H_r, keep_prob=keep_prob)

        return H_r


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, H_r, context_m):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        context_m = tf.cast(context_m, tf.float32)
        # shape -> (b, q, 4n)
        H_r_shape = H_r.get_shape().as_list()
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decoder"):
            W_r = tf.get_variable("V_r", shape=[n_hidden * 4, n_hidden * 2], dtype=dtype, initializer=initializer)

            W_f = tf.get_variable("W_f", shape=[n_hidden * 2, 1], dtype=dtype, initializer=initializer)

            W_h = tf.get_variable("W_h", shape=[n_hidden * 4, n_hidden * 2], dtype=dtype, initializer=initializer)

            B_r = tf.get_variable("B_r", shape=[n_hidden * 2], dtype=dtype)
            B_f = tf.get_variable("B_f", shape=[], dtype=dtype)

            W_r_e = tf.tile(tf.expand_dims(W_r, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_f_e = tf.tile(tf.expand_dims(W_f, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_h_e = tf.tile(tf.expand_dims(W_h, axis=0), multiples=[H_r_shape[0], 1, 1])

            # f1 -> (b, q, 2n)
            f1 = tf.nn.tanh(tf.matmul(H_r, W_r_e) + B_r)
            # s_score -> (b, q, 1)
            s_score = tf.matmul(f1, W_f_e) + B_f
            # s_score -> (b, q)
            s_score = tf.squeeze(s_score, axis=2)

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

            e_prob = tf.nn.softmax(e_score)
            e_prob = tf.multiply(e_prob, context_m)

        return s_score, e_score


class QASystem(object):
    def __init__(self, encoder, decoder, embed_path):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.embed_path = embed_path

        # ==== set up placeholder tokens ========
        self.context = tf.placeholder(tf.int32, shape=(None, context_max_len))
        self.context_m = tf.placeholder(tf.bool, shape=(None, context_max_len))

        self.question = tf.placeholder(tf.int32, shape=(None, question_max_len))
        self.question_m = tf.placeholder(tf.bool, shape=(None, question_max_len))

        self.answer_s = tf.placeholder(tf.int32, shape=(None,))
        self.answer_e = tf.placeholder(tf.int32, shape=(None,))


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        grad_var = self.optimizer.compute_gradients(self.final_loss)
        grad = [i[0] for i in grad_var]
        var = [i[1] for i in grad_var]

        grad, use_norm = tf.clip_by_global_norm(grad, max_grad_norm)

        self.train_op = self.optimizer.apply_gradients(zip(grad, var))

        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        H_r = self.encoder.encode(self.context, self.context_m, self.question, self.question_m, self.embedding)

        self.s_score, self.e_score = self.decoder.decode(H_r, self.context_m)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_s, logits=self.s_score)
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.answer_e, logits=self.e_score)

        self.final_loss = tf.reduce_mean(loss_e + loss_s)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            self.embedding = np.load(self.embed_path)['glove']
            self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, answers, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
