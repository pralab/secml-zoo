from models.tests import CModelsTestCases

from secml.data.loader import CDataLoaderMNIST
from secml.ml.classifiers import CClassifierSVM, CClassifierPyTorch


class TestModelsMNIST(CModelsTestCases):
    """Unittests for MNIST models."""

    @classmethod
    def setUpClass(cls):
        CModelsTestCases.setUpClass()
        # Full MNIST dataset
        cls.ds = CDataLoaderMNIST().load('testing')
        cls.ds.X /= 255
        # MNIST59
        cls.ds_59 = CDataLoaderMNIST().load('testing', digits=(5, 9))
        cls.ds_59.X /= 255
        # MNIST59
        cls.ds_159 = CDataLoaderMNIST().load('testing', digits=(1, 5, 9))
        cls.ds_159.X /= 255

    def test_mnist_svm(self):
        model_id = 'mnist-svm'
        expected_class = CClassifierSVM
        expected_acc = 0.9

        self._test_model(self.ds, model_id, expected_class, expected_acc)

    def test_mnist_cnn(self):
        model_id = 'mnist-cnn'
        expected_class = CClassifierPyTorch
        expected_acc = 0.98

        self._test_model(self.ds, model_id, expected_class, expected_acc)

    def test_mnist59_svm(self):
        model_id = 'mnist59-svm'
        expected_class = CClassifierSVM
        expected_acc = 0.98

        self._test_model(self.ds_59, model_id, expected_class, expected_acc)

    def test_mnist59_svm_rbf(self):
        model_id = 'mnist59-svm-rbf'
        expected_class = CClassifierSVM
        expected_acc = 0.99

        self._test_model(self.ds_59, model_id, expected_class, expected_acc)

    def test_mnist159_cnn(self):
        model_id = 'mnist159-cnn'
        expected_class = CClassifierPyTorch
        expected_acc = 0.99

        self._test_model(self.ds_159, model_id, expected_class, expected_acc)


if __name__ == '__main__':
    CModelsTestCases.main()
