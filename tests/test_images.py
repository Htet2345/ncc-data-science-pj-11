import numpy as np
import pytest
from PIL import Image, ImageDraw
from digit_classifier.images import prepare_image


def test_single_digit_preprocessing_polarity_and_aspect_ratio(tmp_path):
    image = Image.new("L", (80, 100))
    ImageDraw.Draw(image).rectangle((30, 20, 40, 80), fill=255)
    dark_path = tmp_path / "light_digit.png"
    image.save(dark_path)
    light_path = tmp_path / "dark_digit.png"
    Image.fromarray(255 - np.asarray(image)).save(light_path)
    result = prepare_image(dark_path)
    assert result.shape == (1, 28, 28)
    assert result.dtype == np.float32
    assert 0 <= result.min() < result.max() <= 1
    np.testing.assert_array_equal(result, prepare_image(light_path, invert=True))
    rows, cols = np.where(result[0] > .5)
    assert rows.max() - rows.min() + 1 == 20
    assert cols.max() - cols.min() < rows.max() - rows.min()
    assert np.all(result[0, :4] == 0)


@pytest.mark.parametrize("invert", [False, True])
def test_blank_images_rejected(tmp_path, invert):
    path = tmp_path / "blank.png"
    Image.new("L", (28, 28), 255 if invert else 0).save(path)
    with pytest.raises(ValueError, match="blank"):
        prepare_image(path, invert)


def test_transparent_background_composited(tmp_path):
    image = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    ImageDraw.Draw(image).ellipse((10, 5, 25, 35), fill=(255, 255, 255, 255))
    path = tmp_path / "transparent.png"
    image.save(path)
    result = prepare_image(path)
    assert result.max() > .9
    assert result[0, 0, 0] == 0


def test_asymmetric_ink_is_centered_by_pixel_mass(tmp_path):
    image = Image.new("L", (60, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 8, 45, 13), fill=255)
    draw.line((45, 13, 22, 50), fill=255, width=4)
    path = tmp_path / "seven.png"
    image.save(path)
    values = prepare_image(path)[0]
    row_center = np.dot(np.arange(28), values.sum(1)) / values.sum()
    col_center = np.dot(np.arange(28), values.sum(0)) / values.sum()
    assert abs(row_center - 13.5) <= .51
    assert abs(col_center - 13.5) <= .51


def test_missing_image_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_image(tmp_path / "missing.png")
