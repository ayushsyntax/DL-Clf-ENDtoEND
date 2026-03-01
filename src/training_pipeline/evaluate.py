"""Evaluation suite: threshold tuning, metrics, confusion matrix, and JSON reports."""

import json
from pathlib import Path

import numpy
import structlog
import tensorflow
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay
)

from src.common.config import settings

logger = structlog.get_logger()


class ModelEvaluator:
    """Compute and persist full evaluation metrics for the trained binary classifier."""

    def __init__(self, trained_model: tensorflow.keras.Model) -> None:
        self.model = trained_model
        self.output_dir = settings.EVAL_ARTIFACTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation_suite(self, test_ds: tensorflow.data.Dataset) -> dict[str, float]:
        """
        Run full evaluation: AUC, accuracy, precision, recall, optimal threshold.

        Returns:
            Summary metrics dict logged to MLflow and saved as metrics.json.
        """
        logger.info("Running evaluation")
        y_true, y_scores = [], []

        for images, labels in test_ds:
            y_scores.extend(self.model.predict(images, verbose=0).flatten())
            y_true.extend(labels.numpy())

        y_true = numpy.array(y_true)
        y_scores = numpy.array(y_scores)

        best_thresh, best_recall, best_f1 = self._optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= best_thresh).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            "auc": float(roc_auc_score(y_true, y_scores)),
            "accuracy": float(report["accuracy"]),
            "precision": float(report["1"]["precision"]),
            "recall": float(report["1"]["recall"]),
            "best_threshold": best_thresh,
            "best_recall": best_recall,
            "best_f1": best_f1
        }

        self._save_artifacts(confusion_matrix(y_true, y_pred).tolist(), report, metrics)
        self._plot_confusion_matrix(y_true, y_pred)
        logger.info("Evaluation complete", metrics=metrics)
        return metrics

    def _optimal_threshold(
        self, y_true: numpy.ndarray, y_scores: numpy.ndarray
    ) -> tuple[float, float, float]:
        """Find threshold that maximises F1 via precision-recall curve."""
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        idx = int(numpy.argmax(f1))
        best = float(thresh[min(idx, len(thresh) - 1)])
        logger.info("Optimal threshold", threshold=best, f1=float(f1[idx]))
        return best, float(rec[idx]), float(f1[idx])

    def _plot_confusion_matrix(self, y_true: numpy.ndarray, y_pred: numpy.ndarray) -> None:
        """Save confusion matrix PNG to artifacts/plots/."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plot_dir = settings.ARTIFACTS_DIR / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["No Tumor", "Tumor"]
        ).plot()
        plt.savefig(plot_dir / "cm.png")
        plt.close()

    def _save_artifacts(self, cm: list, report: dict, metrics: dict) -> None:
        """Persist confusion matrix, classification report, and metrics as JSON."""
        for path, data in [
            (self.output_dir / "confusion_matrix.json", cm),
            (self.output_dir / "report.json", report),
            (settings.ARTIFACTS_DIR / "metrics.json", metrics)
        ]:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        logger.info("Evaluation artifacts saved", dir=str(self.output_dir))
