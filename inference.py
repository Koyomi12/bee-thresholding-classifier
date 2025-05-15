from enum import Enum
from functools import singledispatch

import numpy as np
from PIL import Image

THRESHOLD_VALUE = 129


class TagStatus(Enum):
    tagged = 0
    untagged = 1


@singledispatch
def classify_image(image, threshold_value: int = THRESHOLD_VALUE):
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@classify_image.register
def _(image: np.ndarray, threshold_value: int = THRESHOLD_VALUE):
    print(image.shape)
    print(threshold_value)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold_value:
                return TagStatus.tagged.name
    return TagStatus.untagged.name


@classify_image.register
def _(cropped_image: Image.Image, threshold_value: int = THRESHOLD_VALUE):
    for i in range(cropped_image.width):
        for j in range(cropped_image.height):
            if cropped_image.getpixel((i, j)) > threshold_value:
                return TagStatus.tagged.name
    return TagStatus.untagged.name
