# PJ11: Handwritten Digit Classification

A TensorFlow/Keras neural network that learns to classify handwritten digits 0–9 using MNIST. It trains locally on CPU, saves a reusable model, evaluates unseen test images, and predicts a digit from a PNG or JPEG.

Course mapping: module **5.5, videos 36–38**, with videos **34–35** as neural-network and TensorFlow prerequisites. This is an original implementation based on the lesson titles; the teacher's exact code, video contents, dataset choices, and submission rubric have not been verified.

## Setup in VS Code (Windows PowerShell)

Open this folder in VS Code. Python 3.13 was used for verification. Create a separate environment, then install the tested versions:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

For compatible dependency ranges instead of exact tested versions, install `requirements-dev.txt`. No GPU or CUDA installation is required. Use **Python: Select Interpreter** to select `.venv\Scripts\python.exe` if VS Code asks. F5 runs training; the included tasks run training or tests. The first TensorFlow import may be slow.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m digit_classifier.train
```

The first run downloads MNIST from the official Keras dataset host (approximately 11 MB). Keras verifies the archive's SHA-256. The archive is cached in this project's `.keras` directory. Internet is required for the first download; later runs use the cache. TensorFlow's own installation is much larger than the dataset.

The original 60,000 training images are shuffled with seed 42 and split into **54,000 training** and **6,000 validation** images. All **10,000 original test images** remain held out until training and validation-based model selection finish. Pixels are converted from 0–255 into float32 values in [0, 1].

Architecture: input 28×28 → flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(10, softmax). Training uses Adam and sparse categorical cross-entropy, up to five epochs, batches of 128, and early stopping on validation loss with the best weights restored. The dense network has 101,770 parameters. Two TensorFlow CPU threads and a small data thread pool limit resource use. The seed supports repeatability; results can still vary across TensorFlow versions and hardware.

Optional settings:

```powershell
.\.venv\Scripts\python.exe -m digit_classifier.train --epochs 8 --batch-size 128 --seed 42 --output artifacts
```

Training writes these local outputs into `artifacts/`:

- `digit_model.keras`: trained neural network; expects normalized 28×28 inputs
- `metrics.json`: test accuracy, macro F1, per-digit precision/recall/F1, confusion matrix, split sizes and training metadata
- `history.json`: training and validation loss/accuracy by epoch
- `learning_curve.png`, `confusion_matrix.png`, `sample_predictions.png`: scientific result charts
- `sample_digits/`: ten labeled images from the test set for trying inference

Generated results are excluded from GitHub. Retrain to reproduce them; no demonstration screenshots are needed.

### Verified run

On 31 October 2026, Python 3.13.1 / TensorFlow 2.21.0 completed five epochs with seed 42. Test accuracy was **97.18%** across the 10,000 held-out MNIST images; macro F1 was **0.9717**. Reloading the saved model reproduced the same accuracy and confusion matrix. All **35 automated tests passed**, and the dependency compatibility check passed.

This accuracy uses the original MNIST test pixels. Imported-image preprocessing is checked separately and can change predictions; the score does not describe accuracy on arbitrary photographs.

## Predict an image

After training, try a saved test image:

```powershell
.\.venv\Scripts\python.exe -m digit_classifier.predict artifacts\sample_digits\test_0_label_7.png
```

For your own single digit written with dark ink on a light background:

```powershell
.\.venv\Scripts\python.exe -m digit_classifier.predict "C:\path\to\digit.png" --invert
```

Default input is light ink on a dark background. `--invert` handles dark ink on a light background. Transparency is composited against the selected background. The foreground is cropped, resized while keeping its aspect ratio to fit 20×20, placed on a black 28×28 canvas, aligned by its center of pixel mass, and normalized. The result shows the predicted digit and the top three model probabilities. Choose the correct polarity; the program does not infer it automatically.

Single-image preprocessing differs slightly from the original MNIST alignment. Phone photos, noisy backgrounds, multiple digits, letters, rotations and handwriting unlike MNIST may perform poorly. A softmax probability is not a calibrated guarantee of correctness. This is a learning project, not a document OCR system.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests run offline. They cover pixel validation, disjoint deterministic splits, image polarity/transparency/aspect ratio, metric arithmetic, invalid inputs, real neural-network training on simple synthetic patterns, and prediction consistency after saving/reloading. The synthetic training test is separate from MNIST evaluation; it does not measure digit accuracy. Full MNIST training is checked separately through the command above.

## Source layout

```text
digit_classifier/
  runtime.py       Project-local cache and TensorFlow CPU configuration
  data.py          MNIST loading, normalization, splits and batched datasets
  model.py         Neural network, inference validation and classification metrics
  images.py        Single-image preprocessing
  plots.py         Headless learning and evaluation charts
  train.py         Training/evaluation command
  predict.py       Single-image prediction command
tests/             Offline automated checks
.vscode/           Interpreter, debugger and task configuration
requirements*.txt  Dependency ranges and exact tested versions
```

## Dataset attribution and references

MNIST is credited to Yann LeCun and Corinna Cortes and derives from NIST data. Keras documents it under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The dataset archive and derived sample images remain local and are not redistributed in this repository.

- [Official Keras MNIST documentation](https://keras.io/api/datasets/mnist/)
- [Keras loader source and download checksum](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py)
- [TensorFlow beginner neural-network tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [TensorFlow installation guide](https://www.tensorflow.org/install/pip)

Repository: [ncc-data-science-pj-11](https://github.com/Htet2345/ncc-data-science-pj-11)
