from process import train_model, extract_pieces_from_image
import sys
import os

SERIALIZATION_FOLDER_PATH = '.'+os.path.sep+'serialized_model'+os.path.sep

VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep
pieces_train_paths = [os.path.join(VALIDATION_DATASET_PATH, image) for image in os.listdir(VALIDATION_DATASET_PATH)]

TEST_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'test'+os.path.sep
pieces_test_paths = [os.path.join(TEST_DATASET_PATH, image) for image in os.listdir(TEST_DATASET_PATH)]

model = train_model(pieces_test_paths, SERIALIZATION_FOLDER_PATH)
pieces_list = extract_pieces_from_image(model, pieces_train_paths)
#print(pieces_test_paths)