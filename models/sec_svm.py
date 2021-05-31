"""
.. module:: Sec-SVM
   :synopsis: Secure Support Vector Machine

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers.secure import CClassifierSecSVM


def sec_svm():
    """Secure Support Vector Machine.

    Algorithm from:

     Demontis et al. "Yes, machine learning can be more secure! a case study
     on android malware detection." IEEE TDSC 2017. https://arxiv.org/abs/1704.08996

    """
    return CClassifierSecSVM()
