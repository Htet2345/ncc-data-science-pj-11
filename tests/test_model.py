import argparse
import numpy as np
import pytest
from digit_classifier.runtime import tf
from digit_classifier.data import dataset
from digit_classifier.model import build_model, classification_metrics, predict_probabilities
from digit_classifier.predict import predict
from digit_classifier.train import positive_int


def test_metric_counts_precision_recall_and_f1():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.eye(10)[[0, 1, 1, 1]]
    result = classification_metrics(labels, probabilities)
    assert result["accuracy"] == .75
    assert result["confusion_matrix"][0][:2] == [1, 1]
    assert result["per_digit"][0]["precision"] == 1
    assert result["per_digit"][0]["recall"] == .5
    assert result["per_digit"][1]["f1"] == pytest.approx(.8)
    assert result["per_digit"][9]["f1"] == 0
    assert result["macro_f1"] == pytest.approx((2 / 3 + .8) / 10)


@pytest.mark.parametrize("labels,probs", [([], np.empty((0, 10))),
    ([10], np.eye(10)[:1]), ([0], np.ones((1, 9))),
    ([0], np.full((1, 10), np.nan)), ([0], np.ones((1, 10))),
    ([0], np.full((1, 10), -.1))])
def test_invalid_metrics_rejected(labels, probs):
    with pytest.raises(ValueError):
        classification_metrics(labels, probs)


@pytest.mark.parametrize("images", [np.zeros((28, 28)), np.empty((0, 28, 28)),
    np.full((1, 28, 28), 255), np.full((1, 28, 28), np.nan)])
def test_prediction_validates_input_before_model_call(images):
    with pytest.raises(ValueError):
        predict_probabilities(None, images)


def test_real_training_save_reload_and_probabilities(tmp_path):
    # Two simple patterns make an offline, fast learnability test, not an MNIST score.
    images = np.zeros((32, 28, 28), dtype=np.float32)
    images[::2, :, :14] = 1
    images[1::2, :, 14:] = 1
    labels = np.tile([0, 1], 16)
    model = build_model(seed=7)
    history = model.fit(dataset(images, labels, 8, True, 7), epochs=3, verbose=0, shuffle=False)
    assert history.history["loss"][-1] < history.history["loss"][0]
    before = predict_probabilities(model, images)
    assert np.mean(before.argmax(1) == labels) >= .95
    path = tmp_path / "test.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    after = predict_probabilities(restored, images)
    np.testing.assert_allclose(after, before, atol=1e-6)
    np.testing.assert_allclose(after.sum(1), 1, atol=1e-6)


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train first"):
        predict(tmp_path / "missing.keras", tmp_path / "digit.png")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_training_settings(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)
