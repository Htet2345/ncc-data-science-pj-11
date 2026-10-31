"""Load official MNIST and split training/validation without using test labels."""
import numpy as np
from .runtime import tf


def normalize(images):
    """Accept uint8-style pixel values; return N x 28 x 28 float32 in [0, 1]."""
    values = np.asarray(images)
    if values.ndim != 3 or values.shape[1:] != (28, 28) or len(values) == 0:
        raise ValueError("Expected a nonempty batch with shape (N, 28, 28).")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Pixels must be numeric.")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
        raise ValueError("Pixels must be finite and between 0 and 255.")
    return values.astype(np.float32) / 255.0


def split_indices(count, validation_size=6000, seed=42):
    if count <= 1 or not 0 < validation_size < count:
        raise ValueError("Validation size must be between 1 and sample count - 1.")
    indices = np.random.default_rng(seed).permutation(count)
    return indices[validation_size:], indices[:validation_size]


def load_mnist():
    """Keras downloads a SHA-256 checked MNIST archive on first use."""
    return tf.keras.datasets.mnist.load_data()


def dataset(images, labels, batch_size=128, training=False, seed=42):
    # A small private thread pool avoids creating a large pool for each epoch.
    result = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        result = result.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return result.batch(batch_size).with_options(options).prefetch(1)
