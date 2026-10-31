import numpy as np
import pytest
from digit_classifier.data import normalize, split_indices


def test_normalization_preserves_shape_and_scales_pixels():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[0, 3, 4] = 255
    images[1, 0, 0] = 128
    result = normalize(images)
    assert result.dtype == np.float32
    assert result.shape == images.shape
    assert result[0, 3, 4] == 1
    assert result[1, 0, 0] == pytest.approx(128 / 255)
    assert images[0, 3, 4] == 255


@pytest.mark.parametrize("images", [np.zeros((28, 28)), np.zeros((1, 10, 10)),
    np.zeros((0, 28, 28)), np.full((1, 28, 28), -1), np.full((1, 28, 28), 256),
    np.full((1, 28, 28), np.nan), np.full((1, 28, 28), np.inf),
    np.full((1, 28, 28), "hello")])
def test_normalization_rejects_invalid_pixels(images):
    with pytest.raises(ValueError):
        normalize(images)


def test_validation_split_is_disjoint_complete_and_repeatable():
    train, validation = split_indices(100, 20, 42)
    repeated = split_indices(100, 20, 42)
    assert len(train) == 80 and len(validation) == 20
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(100))
    assert np.array_equal(train, repeated[0])
    assert not np.array_equal(train, split_indices(100, 20, 43)[0])


@pytest.mark.parametrize("count,size", [(1, 1), (100, 0), (100, 100), (100, -1)])
def test_invalid_split_rejected(count, size):
    with pytest.raises(ValueError):
        split_indices(count, size)
