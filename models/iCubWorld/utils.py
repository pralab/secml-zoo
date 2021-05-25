import numpy as np

from secml.array import CArray
from secml.ml import CNormalizerDNN


from .alexnet import alexnet


def attach_alexnet(clf):
    """Attach AlexNet (as a preprocessor) to input CClassifier.

    The output of `classifier:4` layer is used as input for the classifier.

    """
    clf.preprocess = CNormalizerDNN(net=alexnet(), out_layer='classifier:4')


def ds_numpy_to_pytorch(ds):
    """Converts ds samples from numpy flatten to pytorch flatten."""
    imgs = ds.X
    # Pytorch networks expects images in the tensor format (C x H x W)
    # Our images have been flatten from the numpy format (H x W x C)
    # We firstly need to get back to (n_samples x H x W x C)
    imgs = imgs.tondarray().reshape(
        (imgs.shape[0], ds.header.img_h, ds.header.img_w, 3))
    # Then we move the "C" axis to the correct position,
    # and finally ravel the rows again, done.
    imgs = np.moveaxis(imgs, 3, 1).reshape(
        imgs.shape[0], 3 * ds.header.img_h * ds.header.img_w)
    ds.X = CArray(imgs)
