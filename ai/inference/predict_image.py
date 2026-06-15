import argparse
import json

import cv2

from ai.inference.predictor import predict_image_array


def main():
    parser = argparse.ArgumentParser(
        description="Run local inference against a saved insect classifier artifact."
    )
    parser.add_argument("image_path", help="Image to classify")
    parser.add_argument(
        "--model",
        default="models/insect_classifier.npz",
        help="Saved model artifact path",
    )
    parser.add_argument(
        "--localization-mode",
        choices=("auto", "full"),
        default="auto",
        help="Use auto crop or full image inference",
    )
    parser.add_argument("--min-area-ratio", default=0.002, type=float)
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"image could not be read: {args.image_path}")

    prediction = predict_image_array(
        image,
        artifact_path=args.model,
        localization_mode=args.localization_mode,
        min_area_ratio=args.min_area_ratio,
        use_cache=False,
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
