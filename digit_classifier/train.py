"""Train on MNIST, evaluate once on the held-out test set, and save artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import ROOT, tf
from .data import dataset, load_mnist, normalize, split_indices
from .model import build_model, classification_metrics, predict_probabilities
from .plots import save_plots


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def train(output, epochs=5, batch_size=128, seed=42):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (raw_train, labels), (raw_test, test_labels) = load_mnist()
    train_idx, validation_idx = split_indices(len(labels), 6000, seed)
    images = normalize(raw_train)
    test_images = normalize(raw_test)
    training = dataset(images[train_idx], labels[train_idx], batch_size, True, seed)
    validation = dataset(images[validation_idx], labels[validation_idx], batch_size)
    model = build_model(seed)
    model.summary()
    history = model.fit(training, validation_data=validation, epochs=epochs, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=2, restore_best_weights=True)], verbose=2)
    # All test metrics are computed after training and validation-based selection.
    probabilities = predict_probabilities(model, test_images)
    metrics = classification_metrics(test_labels, probabilities)
    metrics.update({"dataset": "MNIST", "train_samples": len(train_idx),
                    "validation_samples": len(validation_idx), "test_samples": len(test_labels),
                    "seed": seed, "epochs_requested": epochs,
                    "epochs_run": len(history.history["loss"]),
                    "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                    "tensorflow_version": tf.__version__,
                    "test_cross_entropy": float(-np.log(np.clip(
                        probabilities[np.arange(len(test_labels)), test_labels], 1e-7, 1)).mean())})
    model.save(output / "digit_model.keras")
    for name, contents in (("metrics.json", metrics), ("history.json", history.history)):
        (output / name).write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    save_plots(history.history, test_labels, probabilities, test_images, output)
    sample_dir = output / "sample_digits"
    sample_dir.mkdir(exist_ok=True)
    for index in range(10):
        Image.fromarray(raw_test[index]).save(sample_dir / f"test_{index}_label_{test_labels[index]}.png")
    print(f"Test accuracy: {metrics['accuracy']:.2%}; macro F1: {metrics['macro_f1']:.4f}")
    print(f"Artifacts: {output.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.output, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
