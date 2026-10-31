"""A small dense neural network suitable for CPU training."""
import numpy as np
from .runtime import tf


def build_model(seed=42):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="handwritten_digits")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def predict_probabilities(model, images):
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError("Expected a nonempty batch with shape (N, 28, 28).")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Model input must be normalized into [0, 1].")
    # Direct calls avoid tf.data overhead for single-image inference.
    output = np.concatenate([
        np.asarray(model(images[start:start + 256], training=False))
        for start in range(0, len(images), 256)
    ])
    if output.shape != (len(images), 10) or not np.isfinite(output).all():
        raise ValueError("Model must return ten finite probabilities per image.")
    if (output < 0).any() or (output > 1).any() or not np.allclose(output.sum(1), 1, atol=1e-5):
        raise ValueError("Model output must contain normalized probabilities.")
    return output


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, np.arange(10)).all():
        raise ValueError("Expected digit labels 0 through 9.")
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(labels), 10):
        raise ValueError("Expected ten probabilities for each label.")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError("Expected finite normalized probabilities.")
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels.astype(int), predictions), 1)
    per_digit = []
    for digit in range(10):
        true_positive = int(confusion[digit, digit])
        support = int(confusion[digit].sum())
        predicted = int(confusion[:, digit].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        per_digit.append({"digit": digit, "support": support, "precision": precision,
                          "recall": recall, "f1": 2 * precision * recall / (precision + recall)
                          if precision + recall else 0.0})
    return {"accuracy": float(np.mean(predictions == labels)),
            "macro_f1": float(np.mean([row["f1"] for row in per_digit])),
            "per_digit": per_digit, "confusion_matrix": confusion.tolist()}
