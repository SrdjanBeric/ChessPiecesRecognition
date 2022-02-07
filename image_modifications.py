import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin_for_validation(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 20, 255, cv2.THRESH_BINARY)
    return image_bin


def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def split2hsv(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img)
    return h, s, v


def invert(image):
    return 255-image


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def image_to_vector(img):
    return img.flatten()


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def remove_outside_of_contour(img_bin):
    img, contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def remove_inner_contours(image_orig, image_binary):
    img, contours, hierarchy = cv2.findContours(image_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours.remove(contours[0])

    outter_contours = []
    for i in range(len(contours)):
        # print(hierarchy[0, i, 3])
        if (hierarchy[0, i, 3] == -1):
            outter_contours.append(contours[i])

    cv2.drawContours(image_orig, outter_contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    gray = image_gray(image_orig)
    bin = image_bin(gray)
    # display_image(bin)
    return bin


def two_dominant_colors(img):
    height, width, _ = np.shape(img)
    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)
    number_clusters = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(200, 200, row)
        bars.append(bar)
        rgb_values.append(rgb)

    img_bar = np.hstack(bars)

    two_colors = []
    for index, row in enumerate(rgb_values):
        image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                            font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        print(f'{index + 1}. RGB{row}')
        two_colors.append(row)

    print(two_colors)
    # cv2.imshow('Image', img)
    cv2.imshow('Dominant colors', img_bar)
    cv2.waitKey(0)
    return two_colors



def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


def remove_tiles(img, two_colors):
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    color1_mask = cv2.inRange(img, np.array(two_colors[1]), np.array(two_colors[1]))
    img_bin = image_bin(image_gray(img))
    # display_image(color1_mask)
    plt.imshow(img_bin)
    # img_hsv = img_hsv - color1_mask
    # img = img - two_colors[1]
