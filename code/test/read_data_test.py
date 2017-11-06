
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)

path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path += [path + x for x in ['']]


from utils.read_data import mask_dataset, read_raw_answers, read_answers

data_dir = path + "/data/squad"

def test_mask_dataset():
    logging.info("test_mask_dataset")
    dataset = mask_dataset(data_dir, set_names=['train', 'val'], suffixes=['context', 'question'])
    print dataset.keys()
    for k, v in dataset.iteritems():
        print k, len(v), v[0]

def test_read_raw_answers():
    logging.info("test_read_raw_answers")
    raw_answers = read_raw_answers(data_dir)
    print raw_answers.keys()
    for k, v in raw_answers.iteritems():
        print k, len(v), v[0]

def test_read_answers():
    logging.info("test_read_answers")
    answers = read_answers(data_dir)
    print answers.keys()
    for k, v in answers.iteritems():
        print k, len(v), v[0]

if __name__ == '__main__':
    test_mask_dataset()
    test_read_raw_answers()
