import sys
sys.path.insert(0, '../../')

from svm import svm
from secml.data.loader import CDataLoaderMNIST


tr = CDataLoaderMNIST().load('training', digits=(5, 9))
clf = svm()
clf.verbose = 1

clf.fit(tr.X, tr.Y)

state_path = '../mnist59-svm.gz'
clf.save_state(state_path)

import os
print("Model stored into: " + os.path.abspath(state_path))

from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
