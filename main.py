import kagglehub
import matplotlib.image as img
import matplotlib.pyplot    as plt
import numpy 
import random
from enum import Enum


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

# ============================================================
# CLASS CODE
# ============================================================

class SimplePreceptron:
    def __init__(self, 
                 cnt_cells: int, 
                 learning_rate: float = 0.5,
                 max_epochs: int = 100):
        self.w0 = 0
        self.cnt_cells = cnt_cells
        self.cells_w = numpy.zeros(cnt_cells)
        self.learning_rate = learning_rate # 0 < v <= 1 (0.5 - 0.7 is recommended)
        self.iterations = 0
        self.max_epochs = max_epochs
        
        
    # Step 0
    def fill_weigths(self):
        self.w0 = random.uniform(-0.3, 0.3)
        for cell_w in self.cells_w:
            cell_w = random.uniform(-0.3, 0.3)
    
    
    def calc_error(self, exp_y, y) -> float:
        return exp_y - y
    
    
    def threshold_func(self, s) -> float:
        if s >= 0:
            return 1
        else:
            return 0
        
    
    def calc_y(self, x_arr, func_activate) -> float:
        s = self.w0
        for i in range(self.cnt_cells):
            s += x_arr[i] * self.cells_w[i]
        return func_activate(s)
    
    
    def recalc_weigths(self, x_arr: numpy.ndarray, err: float):
        for i in range(self.cnt_cells):
            self.cells_w[i] += err * self.learning_rate * x_arr[i]
    
    
    def train_by_array(self, x_arr: numpy.ndarray, y_exp: float) -> bool:
        is_correct = False
        y = self.calc_y(x_arr, func_activate=self.threshold_func)
        err = self.calc_error(y_exp, y)
        if err != 0:
            is_correct = False
            self.recalc_weigths(x_arr, err)
        else:
            is_correct = True
        return is_correct
            
            
    def fit(self, train_set, name_export):
        correct_cases   = 0
        total_cases     = 0
        
        cases = []
        for i in range(self.max_epochs):
            is_correct = self.train_by_array(train_set[i][0], train_set[i][1])
            if is_correct:
                correct_cases += 1
            total_cases += 1
            cases.append(int(is_correct))
        print(f"Proccent of study: {100*(correct_cases/total_cases)}%")
        self.export_weigths(name_export)
        plt.plot(cases)
        plt.show()
        
        
    
    def export_weigths(self, file_path):
        # Запись в файл (режим 'w' - перезапись)
        numpy.save(file_path, self.cells_w)
        
        # with open(file_path, "w", encoding="utf-8") as file:
        #     file.writelines(self.cells_w)
            
    
    def import_weigths(self, file_path):
        # Дозапись в файл (режим 'a')
        self.cells_w = numpy.load(file_path + ".npy")
        
        
        # with open("my_array.txt", "r", encoding="utf-8") as file:
        #    read_list = file.readlines()
                
        

# ============================================================
# DEBUG CODE
# ============================================================

def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("reevald/geometric-shapes-mathematics")
    print("Path to dataset files:", path)


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


# Prepare train sets
OBJ2TRAIN_FST       = "rectangle"
OBJ2TRAIN_SND       = "triangle"

OBJ2TRAIN_FST_VAL   = 0
OBJ2TRAIN_SND_VAL   = 1

class ObjPrec(Enum):
        RECTANGLE   = OBJ2TRAIN_FST_VAL
        TRIANGLE    = OBJ2TRAIN_SND_VAL

TRAIN_EXAMPLES = 1500
    
def prepare_train_set():
    train_set = []
    print(f"Start to get train sets")
    for i in range(TRAIN_EXAMPLES):
        # Get data
        train_obj1_x = get_x_array_by_path_file(f"dataset/train/{OBJ2TRAIN_FST}/{OBJ2TRAIN_FST}-{i}.jpg")
        train_obj2_x = get_x_array_by_path_file(f"dataset/train/{OBJ2TRAIN_SND}/{OBJ2TRAIN_SND}-{i}.jpg")   
        # Make a list of sets with two values: [X_values, Y_expected]
        train_set.append([train_obj1_x, OBJ2TRAIN_FST_VAL]) 
        train_set.append([train_obj2_x, OBJ2TRAIN_SND_VAL])
        
        if i % 100 == 0:
            print(f"{i} case passed.")
    # Shuffle values between us
    # random.seed(10)
    random.shuffle(train_set)
    return train_set


def test(preceptron: SimplePreceptron, count_tests: int = 100):
    correct = 0
    invalid = 0
    for i in range(count_tests):
        choise = random.randint(0, 1)
        id = random.randint(0, TRAIN_EXAMPLES)
        test_obj_x = []
        test_exp_y = 0
        if choise == OBJ2TRAIN_FST_VAL:
            test_obj_x = get_x_array_by_path_file(f"dataset/train/{OBJ2TRAIN_SND}/{OBJ2TRAIN_SND}-{id}.jpg")
            test_exp_y = OBJ2TRAIN_SND_VAL
        else:
            test_obj_x = get_x_array_by_path_file(f"dataset/train/{OBJ2TRAIN_FST}/{OBJ2TRAIN_FST}-{id}.jpg")
            test_exp_y = OBJ2TRAIN_FST_VAL
        y = preceptron.calc_y(test_obj_x, func_activate=preceptron.threshold_func)
        if y == test_exp_y:
            correct += 1
        else:
            invalid += 1
    
    print(f"Total: {correct}/{correct+invalid} ({(correct/(invalid+correct) * 100)}%)")
        
            
            
        
    
# ============================================================
# MAIN CODE
# ============================================================



train_set = prepare_train_set()
simple_preceptron = SimplePreceptron(224 * 224, learning_rate=0.7, max_epochs=1400)


print(f"Start to fit model")
simple_preceptron.fill_weigths()
simple_preceptron.fit(train_set, name_export="weight-4")

# simple_preceptron.import_weigths("weigth-2")
test(simple_preceptron, 100)