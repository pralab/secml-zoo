from models.tests import CModelsTestCases

from secml.data.splitter import CTrainTestSplit
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.secure import CClassifierSecSVM

from secml.utils import fm, pickle_utils


class TestModelsDrebinRed(CModelsTestCases):
    """Unittests for DrebinRed models."""

    @classmethod
    def setUpClass(cls):
        CModelsTestCases.setUpClass()
        ds = pickle_utils.load(
            fm.join(fm.abspath(__file__),
                    '../../../datasets/DrebinRed/drebin-reduced.tar.gz'))
        _, cls.ts = CTrainTestSplit(train_size=6000, random_state=0).split(ds)

    def test_drebinRed_svm(self):
        model_id = 'drebin-reduced-svm'
        expected_class = CClassifierSVM
        expected_acc = 0.99

        self._test_model(self.ts, model_id, expected_class, expected_acc)

    def test_drebinRed_sec_svm(self):
        model_id = 'drebin-reduced-sec-svm'
        expected_class = CClassifierSecSVM
        expected_acc = 0.99

        self._test_model(self.ts, model_id, expected_class, expected_acc)


if __name__ == '__main__':
    CModelsTestCases.main()
