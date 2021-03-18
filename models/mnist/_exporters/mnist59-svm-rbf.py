import sys
sys.path.insert(0, '../../')

from svm_rbf import svm_rbf
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterKFold
from secml.ml.peval.metrics import CMetricAccuracy


tr = CDataLoaderMNIST().load('training', digits=(5, 9))
tr.X /= 255

clf = svm_rbf()
clf.verbose = 1

xval_params = {'C': [0.1, 1, 10, 100], 'kernel.gamma': [0.001, 0.01, 0.1, 1]}
xval_splitter = CDataSplitterKFold(num_folds=3, random_state=0)
print("Estimating the best training parameters...")
best_params = clf.estimate_parameters(
    dataset=tr,
    parameters=xval_params,
    splitter=xval_splitter,
    metric='accuracy',
)
print("The best training parameters are: ",
      [(k, best_params[k]) for k in sorted(best_params)])

clf.fit(tr.X, tr.Y)

ts = CDataLoaderMNIST().load('testing', digits=(5, 9))
ts.X /= 255

print("Accuracy: {:}".format(
    CMetricAccuracy().performance_score(ts.Y, clf.predict(ts.X))))

state_path = '../mnist59-svm-rbf.gz'
clf.save_state(state_path)

import os
print("Model stored into: " + os.path.abspath(state_path))

from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
