import csv
from pathlib import Path

import cv2
import numpy as np

THRESHOLD_VALUE = 129
TAGGED = "tagged"
UNTAGGED = "untagged"

samples_path = "/home/niklas/Documents/dev/uni/bees/bee-classifier/output/samples.csv"


def main():
    samples = load_samples_csv(Path(samples_path))
    for sample in samples:
        label = label_image(Path(sample["sample_path"]))
        sample[f"pixel_threshold_label_at_{THRESHOLD_VALUE}"] = label
        dictlist_to_csv(samples, samples_path)
    print(evaluate(label, samples))


def find_best_value(imgs_path):
    tagged_img_path = imgs_path / "tagged"
    untagged_img_path = imgs_path / "untagged"

    tagged_imgs = list(tagged_img_path.glob("*.png"))
    untagged_imgs = list(untagged_img_path.glob("*.png"))

    mistakes = []
    for value in range(256):
        mistake_count = 0
        for image_path in tagged_imgs + untagged_imgs:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            tagged = False
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i, j] > value:
                        tagged = True

            if tagged and "/tagged/" not in str(image_path):
                mistake_count += 1
                # print(f"FALSE POSITIVE - {image_path.name}")
            elif not tagged and "/untagged/" not in str(image_path):
                # print(f"FALSE NEGATIVE - {image_path.name}")
                mistake_count += 1
        mistakes.append((value, mistake_count))

    lowest = mistakes[0]
    for value, mistake_count in mistakes:
        if mistake_count < lowest[1]:
            lowest = (value, mistake_count)
    print(lowest)


def create_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # output = np.empty_like(img)
    output = np.empty([50, 50])

    cropped_img = img[100:150, 100:150]
    for i in range(cropped_img.shape[0]):
        for j in range(cropped_img.shape[1]):
            if img[i, j] < THRESHOLD_VALUE:
                output[i, j] = 0
            else:
                output[i, j] = img[i, j]
    # Save the modified image
    cv2.imwrite("modified_image.png", output)


def count_mistakes(imgs_path):
    tagged_img_path = imgs_path / "tagged"
    untagged_img_path = imgs_path / "untagged"

    tagged_imgs = list(tagged_img_path.glob("*.png"))
    untagged_imgs = list(untagged_img_path.glob("*.png"))

    mistake_count = 0
    for image_path in tagged_imgs + untagged_imgs:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        tagged = False
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > THRESHOLD_VALUE:
                    tagged = True

        if tagged and "/tagged/" not in str(image_path):
            mistake_count += 1
            print(f"FALSE POSITIVE - {image_path.name}")
        elif not tagged and "/untagged/" not in str(image_path):
            print(f"FALSE NEGATIVE - {image_path.name}")
            mistake_count += 1
    return mistake_count


def load_samples_csv(path: Path):
    with open(path) as csvfile:
        return list(csv.DictReader(csvfile))


def label_image(image_path: Path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    label = UNTAGGED
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > THRESHOLD_VALUE:
                label = TAGGED
    return label


def evaluate(label, sample_data):
    confusion_matrix = dict(
        true_positive=0, false_negative=0, false_positive=0, true_negative=0
    )
    for sample in sample_data:
        if (
            label == TAGGED
            and sample["manual_evaluation_based_on_first_frame"] == TAGGED
        ):
            confusion_matrix["true_positive"] += 1
        elif (
            label == UNTAGGED
            and sample["manual_evaluation_based_on_first_frame"] == TAGGED
        ):
            confusion_matrix["false_negative"] += 1
        elif (
            label == TAGGED
            and sample["manual_evaluation_based_on_first_frame"] == UNTAGGED
        ):
            confusion_matrix["false_positive"] += 1
        elif (
            label == UNTAGGED
            and sample["manual_evaluation_based_on_first_frame"] == UNTAGGED
        ):
            confusion_matrix["true_negative"] += 1
    return confusion_matrix


def dictlist_to_csv(data: list[dict[str, str]], filename: Path):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(data[0].keys())
        writer.writerows(data.values())


if __name__ == "__main__":
    main()

    # train_imgs = Path(
    #     "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/train/"
    # )
    # find_best_value(train_imgs)

    # print(
    #     count_mistakes(
    #         Path(
    #             "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/validation/"
    #         )
    #     )
    # )

    # print(
    #     count_mistakes(
    #         Path(
    #             "/home/niklas/Documents/dev/uni/bees/bee-classifier/data/cropped/50x50_1/test/"
    #         )
    #     )
    # )
