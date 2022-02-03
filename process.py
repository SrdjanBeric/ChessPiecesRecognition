from image_train_preparation import select_roi
from image_modifications import *

import numpy as bb8
import cv2
import matplotlib.pyplot as plt


def train_model(train_image_paths):
    original_image = cv2.imread(train_image_paths[0])
    gray_image = image_gray(original_image)
    bin_image = image_bin(gray_image)

    org_img, regions = select_roi(original_image, bin_image)

    plt.imshow(org_img)
    plt.show()


    model = 0
    return model