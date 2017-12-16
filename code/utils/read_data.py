"""
read data
"""
import os
import sys
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path += [path + x for x in ['']]

from os.path import join as pjoin
from config import Config as cfg
import numpy as np
import logging
import os


def read_raw_answers(data_dir, set_names=['train', 'val'], suffixes='.answer'):
    '''
    read the raw answers, i.e. the word version not the index one.
    it is used during evaluation.
    return: a dict with
        {'raw_train_answer':[data..],
         'raw_val_answer':[data...]}
    '''
    answers = {}
    for sn in set_names:
        data_path = pjoin(data_dir, sn + suffixes)
        assert os.path.exists(data_path), \
            'the path {} does not exist, please check again.'.format(data_path)
        logging.info('Reading answer from file: {}{}'.format(sn, suffixes))
        with open(data_path, 'r') as fdata:
            raw_answer = [line.strip() for line in fdata.readlines()]
        name = 'raw_' + sn +'_answer'
        answers[name] = raw_answer
    return answers


def read_answers(data_dir, set_names=['train', 'val'], suffix = '.span'):
    #TODO: change the suffix accordingly.
    assert isinstance(set_names, list), 'the type of set_names should be list.'
    assert isinstance(suffix, str), 'the type of set_names should be string.'

    dict = {}
    for sn in set_names:
        data_path = pjoin(data_dir, sn + suffix)
        assert os.path.exists(data_path),\
            'the path {} does not exist, please check again.'.format(data_path)
        logging.info('Reading answer from file: {}{}'.format(sn, suffix))
        with open(data_path, 'r') as fdata:
            answer = [preprocess_answer(line) for line in fdata.readlines()]
        name = sn + '_answer'
        # TODO: need to validate the right way of representing the answer.
        dict[name] = answer

    return dict


def preprocess_answer(string):
    '''
    :param string: one example of answer as a string
    :return:
        reasonable answer where both bounds are in the  range of context length.
    '''
    num = map(int, string.strip().split(' '))
    if min(num) >= cfg.context_max_len:
        # if both answers are greater than cfg.context_max_len
        # cut it at a dummy way.
        # TODO: one change the parameter 10, 50, 0, 20 at will.
        num[0] = np.random.randint(10, cfg.context_max_len - 50)
        num[1] = num[0] + np.random.randint(0, 20)
    elif max(num) >= cfg.context_max_len:
        # if only the upper bound is greater than the context_max_len
        # then make it as context_max_len - 1
        num[1] = cfg.context_max_len - 1
        if num[1] < num[0]:
            num[0] = num[1] - 1
    return num


def mask_dataset(data_dir, set_names=['train', 'val'], suffixes=['context', 'question']):
    '''
    read training and validation dataset: both context and questions.
    return: dataset is a dict with
        {'train_context': [(data, mask),...],
         'train_question': [(data, mask),...],
         'val_context': [(data, mask),...],
         'val_question': [(data,mask),...]}
    '''
    dataset = dict()
    for sn in set_names:
        for sf in suffixes:
            data_path = pjoin(data_dir, sn + '.ids.' + sf)
            logging.info('------------ cute line ----------------')
            logging.info('Reading dataset: {}-{}'.format(sn, sf))

            raw_data = [map(int, line.strip().split(' ')) for line in open(data_path, 'r').readlines()]
            logging.info("the data length is {}".format(len(raw_data)))

            key = sn + '_' + sf
            max_len = cfg.context_max_len if sf == 'context' else cfg.question_max_len
            data_mask = [mask_input(ele, max_len) for ele in raw_data]
            dataset[key] = data_mask
    return dataset


def mask_input(data_list, max_len):
    '''
    return : (data, mask)
    '''
    l = len(data_list)
    mask = [True] * l
    if l > max_len:
        return data_list[:max_len], mask[:max_len]
    else:
        return data_list + [0] * (max_len - l), mask + [False] * (max_len - l)
