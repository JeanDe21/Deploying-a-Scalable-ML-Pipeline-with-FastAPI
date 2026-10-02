import numpy as np

from ml.data import apply_label
from ml.model import compute_model_metrics, train_model


def test_apply_label():
    """Test that binary labels are converted correctly."""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"


def test_train_model():
    """Test that the training function returns a trained model."""
    X_train = np.array([
        [1, 0],
        [2, 0],
        [3, 1],
        [4, 1],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert hasattr(model, "predict")
    assert hasattr(model, "fit")


def test_compute_model_metrics():
    """Test that model metrics are calculated correctly."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == 2 / 3
