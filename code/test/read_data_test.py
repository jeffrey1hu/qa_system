import sys
sys.path.append("..")

from utils.read_data import mask_dataset

def test_mask_dataset():
    data_dir = "../data/squad"
    dataset = mask_dataset(data_dir, set_names=['train', 'val'], suffixes=['context', 'question'])
    print dataset.keys()
    for k, v in dataset.iteritems():
        print k, v[:3]


if __name__ == '__main__':
    test_mask_dataset()