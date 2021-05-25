import sys
sys.path.insert(0, '../../../')

from svm_rbf_ova import svm_rbf_ova
from iCubWorld.utils import attach_alexnet, ds_numpy_to_pytorch

from secml.data.loader import CDataLoaderICubWorld28
from secml.data.splitter import CDataSplitter
from secml.ml.peval.metrics import CMetricAccuracy

dl = CDataLoaderICubWorld28()
dl.verbose = 2

tr = dl.load(
    ds_type='train', day='day4',
    resize_shape=(256, 256), crop_shape=(224, 224),
    normalize=True
)

ds_numpy_to_pytorch(tr)

clf = svm_rbf_ova()
clf.verbose = 1

clf.set_params({
    'C': 1e3,
    'kernel.gamma': 2e-4,
    'class_weight': 'balanced'
})

attach_alexnet(clf)

# xval_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
#                    'kernel.gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]}
#
# xval_splitter = CDataSplitter.create('kfold', num_folds=3, random_state=0)
#
# clf.estimate_parameters(
#     tr, xval_parameters, xval_splitter, 'accuracy')

clf.fit(tr.X, tr.Y)

ts = dl.load(
    ds_type='test', day='day4',
    resize_shape=(256, 256), crop_shape=(224, 224),
    normalize=True
)
ds_numpy_to_pytorch(ts)

print("Accuracy: {:}".format(
    CMetricAccuracy().performance_score(ts.Y, clf.predict(ts.X))))

clf.preprocess = None

state_path = '../icubworld28-day4-svm-rbf.gz'
clf.save_state(state_path)

import os
print("Model stored into: " + os.path.abspath(state_path))

from hashlib import md5
md5_hash = md5()
a_file = open(state_path, "rb")
content = a_file.read()
md5_hash.update(content)

print('md5: ' + md5_hash.hexdigest())
