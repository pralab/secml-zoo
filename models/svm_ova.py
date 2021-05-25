"""
.. module:: SVM-OVA
   :synopsis: Multiclass (One-vs-All) Linear Support Vector Machine

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA


def svm_ova():
    """Multiclass (One-vs-All) Linear Support Vector Machine."""
    return CClassifierMulticlassOVA(CClassifierSVM)
