
'''configurations of the model'''

import os
import tensorflow as tf
from os.path import join as pjoin

class Config:
    # maximum length of context
    context_max_len = 350
    # maximum length of question
    question_max_len = 25

    epochs = 10
    embed_size = 100
    batch_size = 32
    start_lr = 5e-4

    reg = 0.0001

    dtype = tf.float32
    keep_prob = 0.7

    # max_grad_norm = 5.0
    clip_by_val = 10.0

    lstm_num_hidden = 128
    # dataset names
    set_names = ['train', 'val']
    # dataset suffixes
    suffixes = ['context', 'question']

    # absolute path of the root directory.
    ROOT_DIR = os.path.dirname(__file__)
    # data directory
    DATA_DIR = pjoin(ROOT_DIR, 'data', 'squad')

    vocab_file = 'vocab.dat'

    setting_prefix = 'batch_size32keep_prob07reg0001'

    train_dir = 'output/train/{}'.format(setting_prefix)

    summary_dir = 'output/tensorboard/{}/summary_'.format(setting_prefix)

    log_dir = 'output/log/{}'.format(setting_prefix)

    fig_dir = 'output/fig/{}'.format(setting_prefix)

    cache_dir = 'output/cache/{}'.format(setting_prefix)

    # start steps
    start_steps = 0

    # print the loss stat during training
    print_every = 20
    # evaluate sample during test
    sample = 100
    # save checkpoint every n iteration
    save_every = 250
    # save every epoch
    save_every_epoch = True