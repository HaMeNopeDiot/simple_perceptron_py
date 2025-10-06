import matplotlib.image as img
from conversion import make_black_white_matrix, make_x_input_from_matrix


def test():
    file_path = "dataset/train/rectangle/rectangle-0.jpg"
    # f = open(file_path, 'r')

    image = img.imread(file_path)


    # Debug info
    print(image.shape)
    print(f"so: {len(image[0][0])}")
    print(type(image))
    print(len(image))


    # Debug info
    res_m = make_black_white_matrix(image)
    res_x = make_x_input_from_matrix(res_m)
    print(res_m.shape)
    print(type(res_m[0][0]))
    print(res_m[0][0])


    print(f"res_x: {res_x.shape}")