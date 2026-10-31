"""Predict a single handwritten digit from a local PNG/JPEG image."""
import argparse
import json
from pathlib import Path
from PIL import UnidentifiedImageError
from .runtime import ROOT, tf
from .images import prepare_image
from .model import predict_probabilities


def predict(model_path, image_path, invert=False):
    if not Path(model_path).is_file():
        raise FileNotFoundError("Model missing. Run python -m digit_classifier.train first.")
    images = prepare_image(image_path, invert)
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = predict_probabilities(model, images)[0]
    ranking = probabilities.argsort()[::-1][:3]
    return {"digit": int(ranking[0]), "probability": float(probabilities[ranking[0]]),
            "top_3": [{"digit": int(digit), "probability": float(probabilities[digit])}
                      for digit in ranking],
            "note": "Model probabilities are not calibrated confidence guarantees."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "digit_model.keras")
    parser.add_argument("--invert", action="store_true", help="For dark ink on a light background")
    args = parser.parse_args()
    try:
        result = predict(args.model, args.image, args.invert)
    except (FileNotFoundError, ValueError, UnidentifiedImageError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
