import numpy
import matplotlib.image as img


def is_pixel_white(pixel: numpy.ndarray) -> bool:
    for pixel_color in pixel:
        if pixel_color != 0xFF:
            return False
    return True


def make_black_white_matrix(image: numpy.ndarray) -> numpy.ndarray:
    image_w = len(image)
    image_h = len(image[0])
    result_matrix = numpy.zeros((image_w, image_h), dtype=int)
    for i in range(image_h):
        for j in range(image_w):
            pixel = image[i][j]
            result_matrix[i][j] = int(is_pixel_white(pixel))
    return result_matrix


def make_x_input_from_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
    image_w = len(matrix)
    image_h = len(matrix[0])
    result_x = numpy.zeros((image_w * image_h), dtype=int)
    for i in range(image_h):
        for j in range(image_w):
            result_x[j + i * image_w] = matrix[i][j]
    return result_x


def convert_image_to_x_arr(image: numpy.ndarray) -> numpy.ndarray: 
    return make_x_input_from_matrix(make_black_white_matrix(image))


def get_x_array_by_path_file(path: str) -> numpy.ndarray:
    return convert_image_to_x_arr(img.imread(path))