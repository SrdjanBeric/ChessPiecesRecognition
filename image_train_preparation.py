import numpy as bb8
import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_modifications import *
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(50,50), interpolation = cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours.remove(contours[0])
    # cv2.drawContours(image_orig, contours, -1, (255, 0, 0), 1)
    # display_image(image_orig)

    regions_list = []
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100 and h < 130 and h > 100 and w > 20:
            region_bin = image_bin[y:y + h + 1, x:x + w + 1]
            region = image_orig[y:y + h + 1, x:x + w + 1]
            final_region, color = remove_inner_contours(region, region_bin)
            regions_list.append(final_region)
            regions_array.append([resize_region(final_region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display_image(regions_list[1])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    # display_image(sorted_regions[5])
    return image_orig, sorted_regions


def select_roi2(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width, img_channel = image_orig.shape
    tile_height = img_height/8
    tile_width = img_width/8

    piece_colors = []
    coordinates = []
    pieces_list = []
    regions_list = []
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10 and h < tile_height and h > tile_height/2 and w > tile_width/2:
            region_bin = image_bin[y:y + h + 1, x:x + w + 1]
            region = image_orig[y:y + h + 1, x:x + w + 1]
            # display_image(region_bin)
            coordinates.append([x, y])
            final_region, piece_color = remove_inner_contours(region, region_bin)
            piece_colors.append(piece_color)
            regions_list.append(region_bin)
            regions_array.append([resize_region(final_region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display_image(regions_list[7])
    # regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    # display_image(sorted_regions[2])
    # print(coordinates)
    return image_orig, sorted_regions, coordinates, piece_colors


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def prepare_data_for_network(pieces):
    data_for_network = []
    for piece in pieces:
        scaled = scale_to_range(piece)
        data_for_network.append(image_to_vector(scaled))

    return data_for_network


def convert_output():
    pieces = ['rook', 'knight', 'bishop', 'queen', 'king', 'pawn']
    outputs = []
    for i in range(len(pieces)):
        output = bb8.zeros(len(pieces))
        output[i] = 1
        outputs.append(output)
    return bb8.array(outputs)


def train_nn_model(x_train, y_train, serialization_folder):
    trained_model = None
    neural_network = create_network()

    trained_model = train_network(neural_network, x_train, y_train)

    serialize_ann(trained_model, serialization_folder)

    return trained_model


def serialize_ann(nn, serialization_folder):
    print("Saving network...")
    model_json = nn.to_json()
    with open(serialization_folder + "/neuronska.json", "w") as json_file:
        json_file.write(model_json)

    nn.save_weights(serialization_folder + "/neuronska.h5")

    print("Network saved successfully!")


def create_network():
    print("Creating network...")

    neural_network = Sequential()
    neural_network.add(Dense(128, input_dim=2500, activation='sigmoid'))
    neural_network.add(Dense(6, activation='sigmoid'))

    print("Network created successfully!")
    return neural_network


def train_network(neural_network, x_train, y_train):
    print("Training network...")

    x_train = bb8.array(x_train, bb8.float32)
    y_train = bb8.array(y_train, bb8.float32)

    sgd = SGD(lr=0.01, momentum=0.9)
    neural_network.compile(loss='categorical_crossentropy', optimizer=sgd)

    neural_network.fit(x_train, y_train, epochs=20000, batch_size=1, verbose=1, shuffle=False)
    print("Network trained successfully!")
    return neural_network


def display_result(outputs, pieces):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(pieces[winner(output)])
    return result


def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]


def load_trained_model(serialization_folder):
    try:
        print("Loading trained model....")
        json_file = open(serialization_folder + "/neuronska.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        network = model_from_json(loaded_model_json)

        network.load_weights(serialization_folder + "/neuronska.h5")
        print("Trained model found successfully!")
        return network
    except Exception as e:
        print("Warning: No model found!")
        return None


def determine_tiles(img, coordinates):
    img_height, img_width, img_channel = img.shape
    tile_length = img_height/8
    tile_coordinates = []

    i = 0
    while i < len(coordinates):
        coordinates[i][0] = round(coordinates[i][0]/tile_length) + 1
        coordinates[i][1] = round(coordinates[i][1] / tile_length) + 1
        i += 1

    i = 0
    while i < len(coordinates):
        if (coordinates[i][0] == 1):
            coordinates[i][0] = 'a'
        if (coordinates[i][0] == 2):
            coordinates[i][0] = 'b'
        if (coordinates[i][0] == 3):
            coordinates[i][0] = 'c'
        if (coordinates[i][0] == 4):
            coordinates[i][0] = 'd'
        if (coordinates[i][0] == 5):
            coordinates[i][0] = 'e'
        if (coordinates[i][0] == 6):
            coordinates[i][0] = 'f'
        if (coordinates[i][0] == 7):
            coordinates[i][0] = 'g'
        if (coordinates[i][0] == 8):
            coordinates[i][0] = 'h'
        i += 1

    i = 0
    while i < len(coordinates):
        if (coordinates[i][1] == 1):
            coordinates[i][1] = '8'
        if (coordinates[i][1] == 2):
            coordinates[i][1] = '7'
        if (coordinates[i][1] == 3):
            coordinates[i][1] = '6'
        if (coordinates[i][1] == 4):
            coordinates[i][1] = '5'
        if (coordinates[i][1] == 5):
            coordinates[i][1] = '4'
        if (coordinates[i][1] == 6):
            coordinates[i][1] = '3'
        if (coordinates[i][1] == 7):
            coordinates[i][1] = '2'
        if (coordinates[i][1] == 8):
            coordinates[i][1] = '1'
        i += 1

    return coordinates
