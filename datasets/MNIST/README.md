## MNIST: the dataset of handwritten digits

The MNIST database of handwritten digits has a training set of 60,000 examples, 
and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

Four files are available (from http://yann.lecun.com/exdb/mnist/):
- `train-images-idx3-ubyte.gz`  training set images (9912422 bytes)
- `train-labels-idx1-ubyte.gz`  training set labels (28881 bytes)
- `t10k-images-idx3-ubyte.gz`   test set images (1648877 bytes)
- `t10k-labels-idx1-ubyte.gz`   test set labels (4542 bytes)

### How to use
Instructions on how to decode the files are available at: http://yann.lecun.com/exdb/mnist/

While using the [SecML](secml/secml>) library, the `CDataLoaderMNIST` class is available 
to automatically load the dataset in a format (`CDataset`) ready to use.

### Credits
Yann LeCun, Courant Institute, NYU  
Corinna Cortes, Google Labs, New York  
Christopher J.C. Burges, Microsoft Research, Redmond  

[[LeCun et al., 1998a]](
    http://yann.lecun.com/exdb/publis/index.html#lecun-98) Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

### WARNING
This is an integral mirror of the original version of the dataset.
All rights reserved by the original authors.
