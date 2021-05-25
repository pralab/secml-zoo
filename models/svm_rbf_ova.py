"""
.. module:: SVM-RBF-OVA
   :synopsis: Multiclass (One-vs-All) Linear Support Vector Machine

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.kernels import CKernelRBF


def svm_rbf_ova():
    """Multiclass (One-vs-All) Support Vector Machine with RBF Kernel."""
    return CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
