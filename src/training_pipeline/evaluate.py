"""
Evaluation module for classification performance.
Computes matrices and scores on test data.
Persists reports to the local file system.
"""

import json
from pathlib import Path
import tensorflow
import numpy
import structlog
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from src.common.config import settings

logger = structlog.get_logger()


class ModelEvaluator:
    """Audit performance of trained classifier models.

    Attributes:
        model (Model): Trained instance of the classifier.
    """

    def __init__(self, trained_model: tensorflow.keras.Model) -> None:
        """Initialize evaluator with trained model and artifacts.

        Args:
            trained_model: The model to analyze.
        """
        self.model = trained_model
        self.output_directory = settings.EVAL_ARTIFACTS_DIR
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def run_evaluation_suite(
        self,
        testing_dataset: tensorflow.data.Dataset
    ) -> dict[str, float]:
        """Perform full performance analysis.

        Args:
            testing_dataset (Dataset): Data to test against.

        Returns:
            dict[str, float]: Summary of metric scores.
        """
        logger.info("Executing test evaluation")
        ground_truth_labels = []
        probability_predictions = []

        for images, labels in testing_dataset:
            batch_predictions = self.model.predict(images, verbose=0).flatten()
            ground_truth_labels.extend(labels.numpy())
            probability_predictions.extend(batch_predictions)

        y_true = numpy.array(ground_truth_labels)
        y_scores = numpy.array(probability_predictions)

        import os
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(numpy.argmax(f1))
        best_thresh = float(thresh[min(best_idx, len(thresh) - 1)])
        logger.info("threshold_tuned", threshold=best_thresh,
                 recall=float(rec[best_idx]), f1=float(f1[best_idx]))

        y_binary = (y_scores >= best_thresh).astype(int)

        confusion_result = confusion_matrix(y_true, y_binary).tolist()
        detailed_class_report = classification_report(y_true, y_binary, output_dict=True)
        area_under_curve = roc_auc_score(y_true, y_scores)

        self._persist_evaluation_artifacts(confusion_result, detailed_class_report)

        summary_metrics = {
            "auc": float(area_under_curve),
            "accuracy": float(detailed_class_report["accuracy"]),
            "precision": float(detailed_class_report["1"]["precision"]),
            "recall": float(detailed_class_report["1"]["recall"]),
            "best_threshold": best_thresh,
            "test_recall_at_best_thresh": float(rec[best_idx]),
            "test_f1_at_best_thresh": float(f1[best_idx])
        }

        os.makedirs('artifacts/plots', exist_ok=True)
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_binary,
            display_labels=['No Tumor', 'Tumor']
        ).plot()
        plt.savefig('artifacts/plots/cm.png'); plt.close()

        logger.info("Evaluation results summarized", scores=summary_metrics)
        return summary_metrics

    def _persist_evaluation_artifacts(
        self,
        confusion_data: list,
        report_data: dict
    ) -> None:
        """Save raw results as JSON to disk.

        Args:
            confusion_data (list): Matrix data.
            report_data (dict): Mapping to labels and scores.
        """
        matrix_file_path = self.output_directory / "confusion_matrix.json"
        report_file_path = self.output_directory / "report.json"

        with open(matrix_file_path, "w") as file:
            json.dump(confusion_data, file, indent=4)

        with open(report_file_path, "w") as file:
            json.dump(report_data, file, indent=4)

        logger.info("Saved reports", directory=str(self.output_directory))
