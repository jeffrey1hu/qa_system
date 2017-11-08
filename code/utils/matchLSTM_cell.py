import sys
sys.path.append('..')

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

from config import Config as cfg

dtype = cfg.dtype

class matchLSTMcell(RNNCell):

    def __init__(self, input_size, state_size, h_question, h_question_m):
        self.input_size = input_size
        self._state_size = state_size
        self.h_question = h_question
        self.h_question_m = tf.cast(h_question_m, tf.float32)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):

        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            # the batch size
            example_num = tf.shape(inputs)[0]
            initializer = tf.contrib.layers.xavier_initializer()
            # tf.name_scope()
            w_q = tf.get_variable("W_q", shape=[self.input_size, self.input_size], dtype=dtype, initializer=initializer)
            w_p = tf.get_variable("W_p", shape=[self.input_size, self.input_size], dtype=dtype, initializer=initializer)
            w_r = tf.get_variable("W_r", shape=[self.state_size, self.input_size], dtype=dtype, initializer=initializer)
            w_a = tf.get_variable("W_a", shape=[self.input_size, 1], dtype=dtype, initializer=initializer)

            b_p = tf.get_variable("b_p", shape=[self.input_size], dtype=dtype)
            b_a = tf.get_variable("b_a", shape=[1], dtype=dtype)

            # w_q_e -> b * 2n * 2n
            w_q_e = tf.tile(tf.expand_dims(w_q, axis=0), [example_num, 1, 1])

            # g -> b * q * 2n
            g = tf.nn.tanh(tf.matmul(self.h_question, w_q_e)  # shape b * q * 2n
                             + tf.expand_dims(tf.matmul(inputs, w_p) + tf.matmul(state, w_r) + b_p, axis=1)
                             )

            w_a_e = tf.tile(tf.expand_dims(w_a, axis=0), [example_num, 1, 1])

            # alpha -> b * q
            alpha = tf.nn.softmax(tf.squeeze(tf.matmul(g, w_a_e) # shape b * q * 1
                                  + b_a, axis=[2]))
            # mask out the attention over the padding.
            alpha = alpha * self.h_question_m

            # question_attend -> b * 2n
            question_attend = tf.reduce_sum((tf.expand_dims(alpha, [2]) * self.h_question), axis=1)
            # z -> b * 4n
            z = tf.concat([inputs, question_attend], axis=1)

            # with GRU instead
            W_r_gru = tf.get_variable("W_r_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)
            U_r_gru = tf.get_variable("U_r_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)

            # initialize b_r as 1.0 for default "remember"
            b_r_gru = tf.get_variable("b_r_gru",
                                      shape=[self._state_size],
                                      dtype=dtype,
                                      initializer=tf.constant_initializer(1.0))

            W_z_gru = tf.get_variable("W_z_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)

            U_z_gru = tf.get_variable("U_z_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)

            # initialize b_z as 1.0 for default "remember"
            b_z_gru = tf.get_variable("b_z_gru",
                                  shape=[self._state_size],
                                  dtype=dtype,
                                  initializer=tf.constant_initializer(1.0))

            W_o_gru = tf.get_variable("W_o_gru",
                                      shape=[self._state_size, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)

            U_o_gru = tf.get_variable("U_o_gru",
                                      shape=[self.input_size * 2, self._state_size],
                                      dtype=dtype,
                                      initializer=initializer)

            b_o_gru = tf.get_variable("b_o_gru", shape=[self._state_size], dtype=dtype)

            z_t = tf.nn.sigmoid(tf.matmul(z, U_z_gru) + tf.matmul(state, W_z_gru) + b_z_gru)
            r_t = tf.nn.sigmoid(tf.matmul(z, U_r_gru) + tf.matmul(state, W_r_gru) + b_r_gru)

            h_hat = tf.nn.tanh(tf.matmul(z, U_o_gru) + r_t * tf.matmul(state, W_o_gru) + b_o_gru)
            output = z_t * state + (1 - z_t) * h_hat

            new_state = output
        return output, new_state



