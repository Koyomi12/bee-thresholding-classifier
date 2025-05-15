from pathlib import Path

import cv2
import sklearn

from inference import THRESHOLD_VALUE, TagStatus, classify_image
from train import Evaluation

TEST_PATH = Path.cwd() / "data" / "50x50" / "test"


def main():
    total, accuracy = evaluate_test_images(TEST_PATH)
    print(f"Accuracy of the model on the {total} test images: {accuracy:.2f}%")


def evaluate_test_images(image_dir: Path, threshold_value: int = THRESHOLD_VALUE):
    mistake_count = 0
    image_paths = list(image_dir.rglob("*.png"))
    y_true = []
    y_pred = []
    for image_path in image_paths:
        if "/tagged/" in str(image_path):
            y_true.append(TagStatus.tagged.value)
        else:
            y_true.append(TagStatus.untagged.value)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = classify_image(image, threshold_value)
        print(label)
        if label == TagStatus.tagged.name:
            y_pred.append(TagStatus.tagged.value)
        else:
            y_pred.append(TagStatus.untagged.value)
        if (
            label == TagStatus.tagged.name
            and "/tagged/" not in str(image_path)
            or label == TagStatus.untagged.name
            and "/untagged/" not in str(image_path)
        ):
            mistake_count += 1

    # Invert labels because sklearn assumes 1 is positive and 0 is negative result
    y_pred_inversed = [1 if y == TagStatus.tagged.value else 0 for y in y_pred]
    y_true_inversed = [1 if y == TagStatus.tagged.value else 0 for y in y_true]

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
        y_true_inversed, y_pred_inversed
    ).ravel()
    print(f"tp: {tp}")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    f1_score = sklearn.metrics.f1_score(y_true_inversed, y_pred_inversed)
    print(f"F1 score: {f1_score}")
    print(
        sklearn.metrics.classification_report(
            y_true_inversed, y_pred_inversed, digits=2
        )
    )
    total = len(image_paths)
    correct = total - mistake_count
    accuracy = 100 * correct / total
    return Evaluation(total=total, accuracy=accuracy)


if __name__ == "__main__":
    main()
