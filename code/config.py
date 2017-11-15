
'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 400
    # maximum length of question
    question_max_len = 30

    epochs = 5
    embed_size = 100
    batch_size = 128
    start_lr = 5e-4

    reg = 0.001

    dtype = tf.float32
    keep_prob = 0.8

    # max_grad_norm = 5.0
    clip_by_val = 10.0

    lstm_num_hidden = 64
    # dataset names
    set_names = ['train', 'val']
    # dataset suffixes
    suffixes = ['context', 'question']

    # absolute path of the root directory.
    ROOT_DIR = os.path.dirname(__file__)
    # data directory
    DATA_DIR = pjoin(ROOT_DIR, 'data', 'squad')

    vocab_file = 'vocab.dat'

    setting_prefix = 'batch_size128keep_prob08'

    train_dir = 'train/{}'.format(setting_prefix)

    summary_dir = 'summary/{}/summary_'.format(setting_prefix)

    log_dir = 'log/{}'.format(setting_prefix)

    fig_dir = 'fig/{}'.format(setting_prefix)

    cache_dir = 'cache/{}'.format(setting_prefix)

    # print the loss stat during training
    print_every = 40
    # evaluate sample during test
    sample = 100
    # save checkpoint every n iteration
    save_every = 250
    # save every epoch
    save_every_epoch = True