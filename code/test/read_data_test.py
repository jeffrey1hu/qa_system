
import os
import sys
import logging
path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path += [path + x for x in ['']]


from utils.read_data import mask_dataset, read_raw_answers, read_answers

data_dir = path + "/data/squad"
print data_dir


def test_mask_dataset():
    dataset = mask_dataset(data_dir, set_names=['train', 'val'], suffixes=['context', 'question'])
    print dataset.keys()
    for k, v in dataset.iteritems():
        print k, len(v), v[:3]

def test_read_raw_answers():
    raw_answers = read_answers(data_dir)

if __name__ == '__main__':
    test_mask_dataset()