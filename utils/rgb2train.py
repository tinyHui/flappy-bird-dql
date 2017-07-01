import cv2

SIZE = (80, 80)


def __rbg2gray__(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def __shrink_size__(image):
    return cv2.resize(image, SIZE)


def normalize(rgb_image):
    return __rbg2gray__(__shrink_size__(rgb_image))
