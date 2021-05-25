from models.iCubWorld.tests import CICubWorldTestCases

from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.utils import fm, pickle_utils

from models.iCubWorld.utils import ds_numpy_to_pytorch


class TestModelsICubWorld28(CICubWorldTestCases):
    """Unittests for iCubWorld28 models."""

    @classmethod
    def setUpClass(cls):
        CICubWorldTestCases.setUpClass()
        cls.ds = pickle_utils.load(
            fm.join(fm.abspath(__file__), 'iCubWorld28_red.gz'))
        ds_numpy_to_pytorch(cls.ds)

    def test_icubworld28_day4_svm(self):
        model_id = 'icubworld28-day4-svm'
        expected_class = CClassifierMulticlassOVA
        expected_acc = 0.7

        self._test_model(self.ds, model_id, expected_class, expected_acc)

    def test_icubworld28_day4_svm_rbf(self):
        model_id = 'icubworld28-day4-svm-rbf'
        expected_class = CClassifierMulticlassOVA
        expected_acc = 0.69

        self._test_model(self.ds, model_id, expected_class, expected_acc)


if __name__ == '__main__':
    CICubWorldTestCases.main()
