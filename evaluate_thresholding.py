import datetime
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import sklearn
from PIL import Image
from PIL.Image import Image as PILImage
from tqdm import tqdm

from inference import TagStatus, classify_image

TAG_CORRECTOR_DATA = Path("/home/niklas/Documents/dev/uni/bees/bee-data/samples")
# TAG_CORRECTOR_DATA = Path("/mnt/local_storage/processed-bee-data")
ZIPS_PATH = Path("/home/niklas/Documents/dev/uni/bees/bee-data/zipped/")
# ZIPS_PATH = Path("/mnt/trove/wdd/wdd_output_2024/cam0/2024/")
IMAGE_SIZE = 50
FRAMEDIR_WDD = Path("/home/niklas/Documents/dev/uni/bees/bee-data/fullframes")
# FRAMEDIR_WDD = Path("/mnt/trove/wdd/wdd_output_2024/fullframes")


def main():
    """Evaluate Performance of Pixel Intensity Thresholding classifier on full dataset."""
    y_true = []
    y_pred = []
    for zip_path in tqdm(list(ZIPS_PATH.rglob("*.zip"))):
        csv_path = TAG_CORRECTOR_DATA / zip_path.stem / "data.csv"
        data = pd.read_csv(
            csv_path,
            dtype={
                "day_dance_id": "string",
                "waggle_id": "string",
                "category": "Int64",
                "category_label": "string",
                "confidence": "Float64",
                "corrected_category": "Int64",
                "corrected_category_label": "string",
                "dance_type": "string",
                "corrected_dance_type": "string",
            },
            na_filter=False,
        )
        with ZipFile(zip_path) as zip_file:
            video_filenames = list(
                filter(lambda filename: filename.endswith(".apng"), zip_file.namelist())
            )
            for video_filename in tqdm(video_filenames):
                # Find matching metadata file
                metadata_filename = video_filename.replace("frames.apng", "waggle.json")
                with zip_file.open(metadata_filename) as metadata_file:
                    json_data = json.load(metadata_file)
                # We only care about waggles, so filter the rest out. Also,
                # the model thinks the bright pixels of the wooden frame on
                # the comb are tags, so we ignore those detections.
                if json_data["predicted_class_label"] != "waggle" or is_wood_in_frame(
                    json_data
                ):
                    continue
                with zip_file.open(video_filename) as video_file:
                    with Image.open(video_file) as image:
                        cropped_image = crop_center(image, IMAGE_SIZE, IMAGE_SIZE)
                        label = classify_image(cropped_image)
                if label == TagStatus.tagged.name:
                    y_pred.append(TagStatus.tagged.value)
                else:
                    y_pred.append(TagStatus.untagged.value)
                current_sample = data.loc[
                    data["waggle_id"] == str(json_data["waggle_id"])
                ]
                current_sample.reset_index(drop=True, inplace=True)
                if (
                    current_sample.at[0, "category_label"] == "tagged"
                    and current_sample.at[0, "corrected_category_label"] == ""
                ):
                    y_true.append(TagStatus.tagged.value)
                elif (
                    current_sample.at[0, "category_label"] == "tagged"
                    and current_sample.at[0, "corrected_category_label"] == "untagged"
                ):
                    y_true.append(TagStatus.untagged.value)
                elif (
                    current_sample.at[0, "category_label"] == "untagged"
                    and current_sample.at[0, "corrected_category_label"] == ""
                ):
                    y_true.append(TagStatus.untagged.value)
                elif (
                    current_sample.at[0, "category_label"] == "untagged"
                    and current_sample.at[0, "corrected_category_label"] == "tagged"
                ):
                    y_true.append(TagStatus.tagged.value)

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


def is_wood_in_frame(json_data):
    """
    Estimates whether the cropped image of the corresponding dance shows a part
    of the wooden frame on the comb based on the position of the dance.
    """
    df_markers_wdd = pd.read_csv(FRAMEDIR_WDD / "df_markers.csv")
    wdd_markers = get_marker_coordinates_by_timestamp(
        detection_timestamp=json_data["timestamp_begin"],
        df_markers=df_markers_wdd,
    )
    if wdd_markers is None:
        return False

    # Values were determined from a single wdd image.
    wood_offset_x = 100
    wood_offset_top = 80
    wood_offset_bot = 220

    # These borders span an area on the comb that is within the wooden frame.
    border_left = max(
        [wdd_markers[0][0] + wood_offset_x, wdd_markers[2][0] + wood_offset_x]
    )
    border_top = max(
        [wdd_markers[0][1] + wood_offset_top, wdd_markers[1][1] + wood_offset_top]
    )
    border_right = min(
        [wdd_markers[1][0] - wood_offset_x, wdd_markers[3][0] - wood_offset_x]
    )
    border_bot = min(
        [wdd_markers[2][1] - wood_offset_bot, wdd_markers[3][1] - wood_offset_bot]
    )

    center_x, center_y = json_data["roi_center"]
    # Correct the -125 offset of the roi coordinates in the metadata
    correction_offset = 125
    center_x += correction_offset
    center_y += correction_offset

    # Apply offset to account for size of cropped image that the model works on
    frame_offset = IMAGE_SIZE // 2
    return (
        center_x - frame_offset <= border_left
        or center_x + frame_offset >= border_right
        or center_y - frame_offset <= border_top
        or center_y + frame_offset >= border_bot
    )


def crop_center(image: PILImage, output_width: int, output_height: int):
    image_width, image_height = image.size
    left = (image_width - output_width) // 2
    top = (image_height - output_height) // 2
    return image.crop((left, top, left + output_width, top + output_height))


def get_marker_coordinates_by_timestamp(detection_timestamp, df_markers):
    """Gives the most recent marker coordinates from before the dance detection."""
    df_markers["timestamp"] = pd.to_datetime(df_markers["timestamp"])
    marker_timestamps = sorted(df_markers["timestamp"].unique())
    timestamp_to_show = None
    for marker_timestamp in marker_timestamps:
        if marker_timestamp <= datetime.datetime.fromisoformat(detection_timestamp):
            timestamp_to_show = marker_timestamp
    if timestamp_to_show is None:
        return None
    dfsel = df_markers.loc[df_markers["timestamp"] == timestamp_to_show]
    marker_coords = [(row["x"], row["y"]) for _, row in dfsel.iterrows()]
    return marker_coords


if __name__ == "__main__":
    main()
