from image_train_preparation import *
from image_modifications import *

import numpy as bb8
import cv2
import matplotlib.pyplot as plt


def train_model(train_image_paths, serialization_folder):
    original_image = cv2.imread(train_image_paths[0])
    gray_image = image_gray(original_image)
    bin_image = image_bin(gray_image)
    # display_image(bin_image)
    org_img, regions = select_roi(original_image, bin_image)
    # display_image(org_img)
    vector = prepare_for_ann(regions)
    x_train = prepare_data_for_network(vector)
    y_train = convert_output()
    # display_image(original_image)
    model = load_trained_model(serialization_folder)
    if model is None:
        model = train_nn_model(x_train, y_train, serialization_folder)

    # plt.imshow(org_img)
    plt.show()
    return model


def extract_pieces_from_image(trained_model, image_path):
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    img_rgb = load_image(image_path[2])
    # display_image(img_rgb)
    img_bin = image_bin(image_gray(img_rgb))
    h,s,v = split2hsv(img_rgb)
    hs_bin = image_bin_for_validation(h+s)
    hs_bin_opening = cv2.morphologyEx(hs_bin, cv2.MORPH_OPEN, kernel_cross, iterations=1)
    # display_image(hs_bin_opening)
    # hs_bin = erode(hs_bin)
    # display_image(hs_bin)
    org_img, regions, coordinates = select_roi2(img_rgb, hs_bin_opening)
    display_image(org_img)
    coordinates = determine_tiles(org_img, coordinates)

    test_inputs = prepare_for_ann(regions)
    result = trained_model.predict(np.array(test_inputs, np.float32))
    # print(result)
    pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn']
    pieces_results = display_result(result, pieces)
    # print(pieces_results)
    i = 0
    while i < len(pieces_results):
        print(pieces_results[i] + ' ' + coordinates[i][0] + coordinates[i][1])
        i += 1
    # plt.imshow(original_image)
    plt.show()