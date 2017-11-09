# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import time

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from utils.read_data import mask_dataset, read_answers, read_raw_answers
from config import Config as cfg

import logging

logging.basicConfig(level=logging.INFO)

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def print_parameters():
    logging.info('======== trained with key parameters ============')
    logging.info('context max length: {}'.format(cfg.context_max_len))
    logging.info('question max length: {}'.format(cfg.question_max_len))
    logging.info('lstm num hidden: {}'.format(cfg.lstm_num_hidden))
    logging.info('batch size: {}'.format(cfg.batch_size))
    logging.info('start learning rate: {}'.format(cfg.start_lr))
    logging.info('dropout keep probability: {}'.format(cfg.keep_prob))


def main(_):
    set_names = cfg.set_names
    suffixes = cfg.suffixes
    num_hidden = cfg.lstm_num_hidden
    data_dir = cfg.DATA_DIR
    embed_path = pjoin(data_dir, "glove.trimmed." + str(cfg.embed_size) + ".npz")
    vocab_path = pjoin(data_dir, cfg.vocab_file)

    dataset = mask_dataset(data_dir, set_names=set_names, suffixes=suffixes)
    answers = read_answers(data_dir)
    raw_answers = read_raw_answers(data_dir)

    vocab, rev_vocab = initialize_vocab(vocab_path)

    c_time = time.strftime('%Y%m%d_%H%M', time.localtime())
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.cache_dir):
        os.makedirs(cfg.cache_dir)
    if not os.path.exists(cfg.fig_dir):
        os.makedirs(cfg.fig_dir)

    file_handler = logging.FileHandler(pjoin(cfg.log_dir, 'log' + c_time + '.txt'))
    logging.getLogger().addHandler(file_handler)

    print_parameters()

    # gpu setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    encoder = Encoder(size=2 * num_hidden)
    decoder = Decoder(output_size=2 * num_hidden)

    qa = QASystem(encoder, decoder, embed_path)

    with tf.Session(config=config) as sess:

        load_train_dir = get_normalized_train_dir(cfg.train_dir)

        logging.info('=========== trainable varaibles ============')
        for i in tf.trainable_variables():
            logging.info(i.name)
        logging.info('=========== regularized varaibles ============')
        for i in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            logging.info(i.name)

        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(cfg.train_dir)

        qa.train(sess, dataset, answers, save_train_dir, debug_num=5000)
        #
        # qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
