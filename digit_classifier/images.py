"""Convert a tightly cropped single digit image into the MNIST input format."""
import numpy as np
from PIL import Image, ImageOps


def prepare_image(path, invert=False):
    """Expect light ink on a dark background; --invert handles dark-on-light."""
    with Image.open(path) as source:
        # Composite transparency against the background expected before inversion.
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white" if invert else "black")
        background.alpha_composite(rgba)
        image = background.convert("L")
    if invert:
        image = ImageOps.invert(image)
    pixels = np.asarray(image)
    mask = pixels > 30
    if not mask.any():
        raise ValueError("Image is blank; draw one digit or check --invert.")
    rows, cols = np.where(mask)
    image = image.crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))
    # Fit the foreground into 20 x 20, retaining its aspect ratio.
    scale = 20 / max(image.size)
    image = image.resize(tuple(max(1, round(size * scale)) for size in image.size), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    # MNIST aligns the center of pixel mass, not just the bounding-box center.
    values = np.asarray(canvas, dtype=np.float32)
    total = values.sum()
    row_mass = values.sum(axis=1)
    col_mass = values.sum(axis=0)
    row_center = float(np.dot(np.arange(28), row_mass) / total)
    col_center = float(np.dot(np.arange(28), col_mass) / total)
    dx = round(13.5 - col_center)
    dy = round(13.5 - row_center)
    centered = Image.new("L", (28, 28))
    centered.paste(canvas, (dx, dy))
    return np.asarray(centered, dtype=np.float32)[None, :, :] / 255.0
