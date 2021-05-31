from models.tests import CModelsTestCases

from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.secure import CClassifierSecSVM

from secml.utils import fm, pickle_utils


class TestModelsDrebin(CModelsTestCases):
    """Unittests for Drebin models."""

    @classmethod
    def setUpClass(cls):
        CModelsTestCases.setUpClass()
        cls.ds = pickle_utils.load(
            fm.join(fm.abspath(__file__), 'drebin_red.gz'))

    def test_drebin_svm(self):
        model_id = 'drebin-svm'
        expected_class = CClassifierSVM
        expected_acc = 0.99

        self._test_model(self.ds, model_id, expected_class, expected_acc)

    def test_drebin_sec_svm(self):
        model_id = 'drebin-sec-svm'
        expected_class = CClassifierSecSVM
        expected_acc = 0.98

        self._test_model(self.ds, model_id, expected_class, expected_acc)


if __name__ == '__main__':
    CModelsTestCases.main()
