from collections import namedtuple
from pathlib import Path

import cv2

from inference import TagStatus, classify_image

Evaluation = namedtuple("Evaluation", ["total", "accuracy"])
ThresholdEvaluation = namedtuple(
    "ThresholdEvaluation", ("threshold_value",) + Evaluation._fields
)

TRAIN_PATH = Path.cwd() / "data" / "50x50" / "train"
VALIDATION_PATH = Path.cwd() / "data" / "50x50" / "validation"


def main() -> None:
    threshold_value, total, accuracy = find_best_threshold_value(
        train_image_dir=TRAIN_PATH, validation_image_dir=VALIDATION_PATH
    )
    print(
        f"Best accuracy of the model on the {total} training and validation images using a threshold value of {threshold_value}: {accuracy:.2f}%"
    )


def find_best_threshold_value(
    train_image_dir: Path, validation_image_dir: Path
) -> ThresholdEvaluation:
    """Searches for the threshold value with the best accuracy."""
    train_image_paths = list(train_image_dir.rglob("*.png"))
    validation_image_paths = list(validation_image_dir.rglob("*.png"))
    image_paths = train_image_paths + validation_image_paths
    mistakes = []
    for threshold_value in range(256):
        mistake_count = 0
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            label = classify_image(image, threshold_value)
            if label == TagStatus.tagged.name and "/tagged/" not in str(image_path):
                mistake_count += 1
            elif label == TagStatus.untagged.name and "/untagged/" not in str(
                image_path
            ):
                mistake_count += 1
        mistakes.append((threshold_value, mistake_count))

    lowest = mistakes[0]
    for threshold_value, mistake_count in mistakes:
        if mistake_count < lowest[1]:
            lowest = (threshold_value, mistake_count)
    total = len(image_paths)
    correct = total - lowest[1]
    accuracy = 100 * correct / total
    return ThresholdEvaluation(
        threshold_value=lowest[0], total=total, accuracy=accuracy
    )


if __name__ == "__main__":
    main()
