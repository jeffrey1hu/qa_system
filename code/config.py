
'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30
    n_hidden = 100
    dtype = tf.float32
    keep_prob = 0.8
    learning_rate = 0.001
    max_grad_norm = 5.0
    # dataset names
    set_names = ['train', 'val']
    # dataset suffixes
    suffixes = ['context', 'question']