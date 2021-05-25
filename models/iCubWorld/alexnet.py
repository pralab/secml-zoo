"""
.. module:: AlexNet
   :synopsis: AlexNet Convolutional Neural Network

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from torchvision import models

from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features.normalization import CNormalizerMeanStd


def alexnet():
    """CClassifierPyTorch with AlexNet Convolutional Neural Network."""
    model = models.alexnet(pretrained=True)

    norm_rgb = CNormalizerMeanStd((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    net = CClassifierPyTorch(model=model,
                             input_shape=(3, 224, 224),
                             pretrained=True,
                             softmax_outputs=False,
                             preprocess=norm_rgb,
                             random_state=0)

    return net
