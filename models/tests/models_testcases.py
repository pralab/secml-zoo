from secml.testing import CUnitTest

from secml.model_zoo import load_model
from secml.ml.peval.metrics import CMetricAccuracy


class CModelsTestCases(CUnitTest):
    """Unittests interface for models."""

    def _test_model(self, ds, model_id, expected_class, expected_acc):
        """Test the pretrained model class and accuracy.

        Parameters
        ----------
        model_id : str
            Identifier of the model to load.
        expected_class : CClassifier
            Expected class of the model.
        expected_acc : float
            Expected accuracy of the model on the test dataset.
            We check if the computed accuracy is higher than this value.

        """
        clf = load_model(model_id)
        self.logger.info("Loaded {:}:\n{:}".format(model_id, clf))

        self.assertIsInstance(clf, expected_class)

        y_pred = clf.predict(ds.X)
        acc = CMetricAccuracy().performance_score(ds.Y, y_pred)

        self.logger.info(
            "Accuracy: {:} (expected {:})".format(acc, expected_acc))
        self.assertGreater(acc, expected_acc)
