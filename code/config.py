
'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30

    epochs = 3
    embed_size = 100
    batch_size = 32
    start_lr = 5e-4
    n_hidden = 100
    dtype = tf.float32
    keep_prob = 0.8
    max_grad_norm = 5.0
    lstm_num_hidden = 100
    # dataset names
    set_names = ['train', 'val']
    # dataset suffixes
    suffixes = ['context', 'question']

    # absolute path of the root directory.
    ROOT_DIR = os.path.dirname(__file__)
    # data directory
    DATA_DIR = pjoin(ROOT_DIR, 'data', 'squad')

    vocab_file = 'vocab.dat'

    train_dir = 'train/ckpt'

    summary_dir = 'summary/summary_'

    log_dir = 'log'

    fig_dir = 'fig'

    cache_dir = 'cache'

    # print the loss stat during training
    print_every = 100
    # evaluate sample during test
    sample = 100
    # save checkpoint every n iteration
    save_every = 2000
    # save every epoch
    save_every_epoch = True