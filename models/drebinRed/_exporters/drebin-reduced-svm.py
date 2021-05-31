import sys
sys.path.insert(0, '../../')

from svm import svm

from secml.data import CDataset
from secml.data.splitter import CTrainTestSplit, CDataSplitter
from secml.ml.peval.metrics import CMetricAccuracy

ds = CDataset.load('../../../datasets/DrebinRed/drebin-reduced.tar.gz')

tr, ts = CTrainTestSplit(train_size=6000, random_state=0).split(ds)

clf = svm()
clf.verbose = 1

clf.set_params({
    'C': 0.1,
})

# xval_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]}
#
# xval_splitter = CDataSplitter.create('kfold', num_folds=3, random_state=0)
#
# clf.estimate_parameters(
#     tr, xval_parameters, xval_splitter, 'accuracy')

clf.fit(tr.X, tr.Y)

print("Accuracy: {:}".format(
    CMetricAccuracy().performance_score(ts.Y, clf.predict(ts.X))))

state_path = '../drebin-reduced-svm.gz'
clf.save_state(state_path)

import os
print("Model stored into: " + os.path.abspath(state_path))

from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
