import sys
sys.path.insert(0, '../../')

from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterShuffle

from models.mnist.mnist_cnn import mnist_cnn


tr_data = CDataLoaderMNIST().load(ds='training')
tr_data.X /= 255.0
splitter = CDataSplitterShuffle(num_folds=1, train_size=0.5,
                                test_size=0.5, random_state=1)
splitter.compute_indices(tr_data)
tr_set = tr_data[splitter.ts_idx[0], :]

clf = mnist_cnn(random_state=1)
clf.fit(tr_set)

state_path = '../mnist-cnn.gz'
clf.save_state(state_path)

import os
print("Model stored into: " + os.path.abspath(state_path))

from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
