# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
import time
import logging

import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import variable_scope as vs

from tqdm import tqdm

from evaluate import exact_match_score, f1_score
from utils.matchLSTM_cell import matchLSTMcell
from config import Config as cfg

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from os.path import join as pjoin

logging.basicConfig(level=logging.INFO)

context_max_len = cfg.context_max_len
question_max_len = cfg.question_max_len
n_hidden = cfg.lstm_num_hidden
dtype = cfg.dtype
keep_prob = cfg.keep_prob
start_lr = cfg.start_lr
# max_grad_norm = cfg.max_grad_norm
regularizer = tf.contrib.layers.l2_regularizer(cfg.reg)
clip_by_val = cfg.clip_by_val

def smooth(a, beta=0.8):
    '''smooth the curve'''

    for i in xrange(1, len(a)):
        a[i] = beta * a[i-1] + (1 - beta)*a[i]
    return a


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


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



def softmax_mask_prepro(tensor, mask):  # set huge neg number(-1e10) in padding area
    assert tensor.get_shape().ndims == mask.get_shape().ndims
    m0 = tf.subtract(tf.constant(1.0), tf.cast(mask, 'float32'))
    paddings = tf.multiply(m0, tf.constant(-1e10))
    tensor = tf.where(mask, tensor, paddings)
    return tensor


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
        context_embed = tf.nn.dropout(context_embed, keep_prob=keep_prob)
        question_embed = tf.nn.embedding_lookup(embedding, question)
        question_embed = tf.nn.dropout(question_embed, keep_prob=keep_prob)

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
            variable_summaries(H_context)

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
            variable_summaries(H_question)


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
            variable_summaries(H_r)

        return H_r


class Decoder(object):
    def __init__(self, output_size=2 * n_hidden):
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
        context_m_float = tf.cast(context_m, tf.float32)
        # shape -> (b, q, 4n)
        H_r_shape = tf.shape(H_r)
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("decoder"):
            W_r = tf.get_variable("V_r", shape=[n_hidden * 4, n_hidden * 2],
                                  dtype=dtype, initializer=initializer, regularizer=regularizer)

            W_f = tf.get_variable("W_f", shape=[n_hidden * 2, 1],
                                  dtype=dtype, initializer=initializer, regularizer=regularizer)

            W_h = tf.get_variable("W_h", shape=[n_hidden * 4, n_hidden * 2],
                                  dtype=dtype, initializer=initializer, regularizer=regularizer)

            B_r = tf.get_variable("B_r", shape=[n_hidden * 2], dtype=dtype, initializer=tf.zeros_initializer())
            B_f = tf.get_variable("B_f", shape=[], dtype=dtype, initializer=tf.zeros_initializer())

            W_r_e = tf.tile(tf.expand_dims(W_r, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_f_e = tf.tile(tf.expand_dims(W_f, axis=0), multiples=[H_r_shape[0], 1, 1])
            W_h_e = tf.tile(tf.expand_dims(W_h, axis=0), multiples=[H_r_shape[0], 1, 1])

            # f1 -> (b, q, 2n)
            f1 = tf.nn.tanh(tf.matmul(H_r, W_r_e) + B_r)
            f1 = tf.nn.dropout(f1, keep_prob=keep_prob)
            with tf.name_scope('starter_score'):
                # s_score -> (b, q, 1)
                s_score = tf.matmul(f1, W_f_e) + B_f
                # s_score -> (b, q)
                s_score = tf.squeeze(s_score, axis=2)
                s_score = softmax_mask_prepro(s_score, context_m)
                variable_summaries(s_score)

            with tf.name_scope('starter_prob'):
                # the prob distribution of start index
                s_prob = tf.nn.softmax(s_score)
                s_prob = s_prob * context_m_float
                variable_summaries(s_prob)
            # Hr_attend -> (batch_size, 4n)
            Hr_attend = tf.reduce_sum(H_r * tf.expand_dims(s_prob, axis=2), axis=1)

            # f2 = tf.nn.tanh(tf.matmul(H_r, W_r_e)
            #                 + tf.matmul(tf.tile(tf.expand_dims(Hr_attend, axis=1), multiples=[1, H_r_shape[1], 1]), W_h_e)
            #                 + B_r)
            f2 = tf.nn.tanh(tf.matmul(H_r, W_r_e)
                            + tf.expand_dims(tf.matmul(Hr_attend, W_h), axis=1)
                            + B_r)
            with tf.name_scope('end_score'):
                e_score = tf.matmul(f2, W_f_e) + B_f
                e_score = tf.squeeze(e_score, axis=2)
                e_score = softmax_mask_prepro(e_score, context_m)
                variable_summaries(e_score)

            with tf.name_scope('end_prob'):
                e_prob = tf.nn.softmax(e_score)
                e_prob = tf.multiply(e_prob, context_m_float)
                variable_summaries(e_prob)

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

            self.global_step = tf.Variable(cfg.start_steps, trainable=False)
            self.starter_learning_rate = tf.placeholder(tf.float32, name='start_lr')
            learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                       1000, 0.9, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            # grad_var = self.optimizer.compute_gradients(self.final_loss)
            # grad = [i[0] for i in grad_var]
            # var = [i[1] for i in grad_var]
            # self.grad_norm = tf.global_norm(grad)
            # tf.summary.scalar('grad_norm', self.grad_norm)
            # grad, use_norm = tf.clip_by_global_norm(grad, max_grad_norm)
            #
            # self.train_op = self.optimizer.apply_gradients(zip(grad, var), global_step=self.global_step)

            gradients = self.optimizer.compute_gradients(self.final_loss)
            capped_gvs = [(tf.clip_by_value(grad, -clip_by_val, clip_by_val), var) for grad, var in gradients]
            grad = [x[0] for x in gradients]
            self.grad_norm = tf.global_norm(grad)
            tf.summary.scalar('grad_norm', self.grad_norm)
            self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

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

            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.final_loss = tf.reduce_mean(loss_e + loss_s) + reg_term
        tf.summary.scalar('final_loss', self.final_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        logging.info('embed size: {} for path {}'.format(cfg.embed_size, self.embed_path))
        with vs.variable_scope("embeddings"):
            self.embedding = np.load(self.embed_path)['glove']
            self.embedding = tf.Variable(self.embedding, dtype=tf.float32, trainable=False)

    def optimize(self, session, context, question, answer, lr):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.context] = context[:, 0, :]
        input_feed[self.context_m] = context[:, 1, :]
        input_feed[self.question] = question[:, 0, :]
        input_feed[self.question_m] = question[:, 1, :]
        input_feed[self.answer_s] = answer[:, 0]
        input_feed[self.answer_e] = answer[:, 1]
        input_feed[self.starter_learning_rate] = lr

        output_feed = [self.merged, self.train_op, self.final_loss, self.grad_norm]

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

    def decode(self, session, context, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}
        input_feed[self.context] = context[:, 0, :]
        input_feed[self.context_m] = context[:, 1, :]
        input_feed[self.question] = question[:, 0, :]
        input_feed[self.question_m] = question[:, 1, :]

        output_feed = [self.s_score, self.e_score]

        s_score, e_score = session.run(output_feed, input_feed)

        return s_score, e_score

    def answer(self, session, context, question):

        yp, yp2 = self.decode(session, context, question)

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


    def evaluate_answer(self, session, dataset, raw_answers, rev_vocab,
                        sample=(100, 100), log=False, training=False):
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
        if not isinstance(rev_vocab, np.ndarray):
            rev_vocab = np.array(rev_vocab)

        if not isinstance(sample, tuple):
            sample = (sample, sample)

        tf1 = 0.
        tem = 0.

        input_batch_size = 100

        if training:
            train_len = sample[0]

            train_context = dataset['train_context'][:train_len]
            train_question = dataset['train_question'][:train_len]
            train_answer = raw_answers['raw_train_answer'][:train_len]


            train_a_s = np.array([], dtype=np.int32)
            train_a_e = np.array([], dtype=np.int32)

            for i in tqdm(range(train_len // input_batch_size), desc='trianing set'):
                train_as, train_ae = self.answer(session,
                                                 np.array(train_context[i * input_batch_size:(i + 1) * input_batch_size]),
                                                 np.array(train_question[i * input_batch_size:(i + 1) * input_batch_size]))

                train_a_s = np.concatenate((train_a_s, train_as), axis=0)
                train_a_e = np.concatenate((train_a_e, train_ae), axis=0)

            # a_s and a_e -> (sample_num)
            for i in range(train_len):
                prediction_ids = train_context[i][0][train_a_s[i]:train_a_e[i]+1]
                prediction_answer = ' '.join(rev_vocab[prediction_ids])
                raw_answer = train_answer[i]
                tf1 += f1_score(prediction_answer, raw_answer)
                tem += exact_match_score(prediction_answer, raw_answer)
                # if i < 10:
                #     print("predict_answer: ", prediction_answer)
                #     print("ground truth: ", raw_answer)
                #     print ("f1: ", f1_score(prediction_answer, raw_answer))

            if log:
                logging.info("Training set ==> F1: {}, EM: {}, for {} samples".
                             format(tf1 / train_len, tem / train_len, train_len))

        f1 = 0.
        em = 0.
        val_len = sample[1]

        val_context = dataset['val_context'][:val_len]
        val_question = dataset['val_question'][:val_len]
        val_answer = raw_answers['raw_val_answer'][:val_len]

        val_a_s = np.array([], dtype=np.int32)
        val_a_e = np.array([], dtype=np.int32)

        for i in tqdm(range(val_len // input_batch_size), desc='val set'):
            val_as, val_ae = self.answer(session,
                                         np.array(val_context[i * input_batch_size:(i + 1) * input_batch_size]),
                                         np.array(val_question[i * input_batch_size:(i + 1) * input_batch_size]))

            val_a_s = np.concatenate((val_a_s, val_as), axis=0)
            val_a_e = np.concatenate((val_a_e, val_ae), axis=0)

        # a_s and a_e -> (sample_num)
        for i in range(val_len):
            prediction_ids = val_context[i][0][val_a_s[i]:val_a_e[i]+1]
            prediction_answer = ' '.join(rev_vocab[prediction_ids])
            raw_answer = val_answer[i]
            f1 += f1_score(prediction_answer, raw_answer)
            em += exact_match_score(prediction_answer, raw_answer)
            # if i < 10:
            #     print("predict_answer: ", prediction_answer)
            #     print("ground truth: ", raw_answer)
            #     print ("f1: ", f1_score(prediction_answer, raw_answer))

        if log:
            logging.info("val set ==> F1: {}, EM: {}, for {} samples".
                         format(f1 / val_len, em / val_len, val_len))

        if training:
            return tf1/train_len, tem/train_len, f1/val_len, em/val_len
        else:
            return f1/val_len, em/val_len


    def train(self, session, dataset, answers, train_dir, raw_answers, rev_vocab, debug_num=None):
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

        # train_context -> (num, 2, max_length)
        train_context = np.array(dataset['train_context'])
        train_question = np.array(dataset['train_question'])
        # train_answer -> (num, 2)
        train_answer = np.array(answers['train_answer'])

        print_every = cfg.print_every

        if debug_num:
            assert isinstance(debug_num, int), 'the debug number should be a integer'
            assert debug_num < len(train_answer), 'check debug number!'
            train_answer = train_answer[0:debug_num]
            train_context = train_context[0:debug_num]
            train_question = train_question[0:debug_num]
            print_every = 5

        num_example = len(train_answer)
        logging.info('num example is {}'.format(num_example))
        shuffle_list = np.arange(num_example)

        self.epochs = cfg.epochs

        self.losses = []

        self.norms = []
        self.train_evals = []
        self.val_evals = []
        self.iters = cfg.start_steps
        save_path = pjoin(train_dir, 'weights')

        self.train_writer = tf.summary.FileWriter(cfg.summary_dir + str(start_lr),
                                                  session.graph)

        batch_size = cfg.batch_size
        batch_num = int(num_example / batch_size)
        total_iterations = self.epochs * batch_num + cfg.start_steps

        tic = time.time()

        for ep in range(self.epochs):
            np.random.shuffle(shuffle_list)

            train_context = train_context[shuffle_list]
            train_question = train_question[shuffle_list]
            train_answer = train_answer[shuffle_list]

            logging.info('training epoch ---- {}/{} -----'.format(ep + 1, self.epochs))
            ep_loss = 0.

            for it in xrange(batch_num):
                sys.stdout.write('> %d / %d \r' % (self.iters % print_every, print_every))
                sys.stdout.flush()

                context = train_context[it * batch_size: (it + 1) * batch_size]
                question = train_question[it * batch_size: (it + 1) * batch_size]
                answer = train_answer[it * batch_size: (it + 1) * batch_size]

                outputs = self.optimize(session, context, question, answer, start_lr)

                self.train_writer.add_summary(outputs[0], self.iters)
                loss, grad_norm = outputs[2:]

                ep_loss += loss
                self.losses.append(loss)
                self.norms.append(grad_norm)
                self.iters += 1

                if self.iters % print_every == 0:
                    toc = time.time()
                    logging.info('iters: {}/{} loss: {} norm: {}. time: {} secs'.format(
                        self.iters, total_iterations, loss, grad_norm, toc - tic))
                    tf1, tem, f1, em = self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                                            training=True, log=True, sample=cfg.sample)
                    self.train_evals.append((tf1, tem))
                    self.val_evals.append((f1, em))
                    tic = time.time()

                if self.iters % cfg.save_every == 0:
                    self.saver.save(session, save_path, global_step=self.iters)
                    self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                         training=True, log=True, sample=4000)
            if cfg.save_every_epoch:
                self.saver.save(session, save_path, global_step=self.iters)
                self.evaluate_answer(session, dataset, raw_answers, rev_vocab,
                                     training=True, log=True, sample=4000)
            logging.info('average loss of epoch {}/{} is {}'.format(ep + 1, self.epochs, ep_loss / batch_num))

            data_dict = {'losses': self.losses, 'norms': self.norms,
                         'train_eval': self.train_evals, 'val_eval': self.val_evals}
            c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
            data_save_path = pjoin('cache', str(self.iters) + 'iters' + c_time + '.npz')
            np.savez(data_save_path, data_dict)
            self.draw_figs(c_time, start_lr)

            # plt.show()

    def draw_figs(self, c_time, lr):
        '''draw figs'''
        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2, 1, 1)
        plt.plot(smooth(self.losses))
        plt.xlabel('iterations')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(smooth(self.norms))
        plt.xlabel('iterations')
        plt.ylabel('gradients norms')
        plt.title('lr={}'.format(lr))
        fig.tight_layout()

        output_fig = 'lr-' + str(lr) + 'loss-norms' + c_time + '.pdf'
        plt.savefig(pjoin(cfg.fig_dir, output_fig), format='pdf')

        # plt.figure()
        fig, _ = plt.subplots(nrows=2, ncols=1)
        plt.subplot(2, 1, 1)
        plt.plot(smooth([x[0] for x in self.train_evals]))
        plt.plot(smooth([x[0] for x in self.val_evals]))
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('f1 score')

        plt.subplot(2, 1, 2)
        plt.plot([x[1] for x in self.train_evals])
        plt.plot([x[1] for x in self.val_evals])
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iterations')
        plt.ylabel('em score')
        plt.title('lr={}'.format(lr))
        fig.tight_layout()

        eval_out = 'lr-' + str(lr) + 'f1-em' + c_time + '.pdf'
        plt.savefig(pjoin(cfg.fig_dir, eval_out), format='pdf')
