import random
import numpy
import matplotlib.pyplot    as plt
import os

from conversion import get_x_array_by_path_file
from utility    import TRAIN_EXAMPLES, OBJ2TRAIN_SND, OBJ2TRAIN_SND_VAL,\
    OBJ2TRAIN_FST, OBJ2TRAIN_FST_VAL, DEFAULT_W_DIR, DEFAULT_OBJ_TRAIN_DATASET_PATH
    
from utility import TEST_OFFSET, TEST_EXAMPLES, ObjType



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
        os.makedirs(DEFAULT_W_DIR, exist_ok=True)
        numpy.save(DEFAULT_W_DIR + "/" + file_path, self.cells_w)
        
    
    def import_weigths(self, file_path):
        os.makedirs(DEFAULT_W_DIR, exist_ok=True)
        self.cells_w = numpy.load(DEFAULT_W_DIR + "/" + file_path + ".npy")
        
        
    def test(self, count_tests: int = 100):
        correct = 0
        invalid = 0
        for i in range(count_tests):
            choise = random.randint(0, 1)
            id = random.randint(0, TEST_EXAMPLES) + TEST_OFFSET
            test_obj_x = []
            test_exp_y = 0
            if choise == OBJ2TRAIN_FST_VAL:
                test_obj_x = get_x_array_by_path_file(f"{DEFAULT_OBJ_TRAIN_DATASET_PATH}/test/{OBJ2TRAIN_SND}/{OBJ2TRAIN_SND}-{id}.jpg")
                test_exp_y = OBJ2TRAIN_SND_VAL
            else:
                test_obj_x = get_x_array_by_path_file(f"{DEFAULT_OBJ_TRAIN_DATASET_PATH}/test/{OBJ2TRAIN_FST}/{OBJ2TRAIN_FST}-{id}.jpg")
                test_exp_y = OBJ2TRAIN_FST_VAL
            y = self.calc_y(test_obj_x, func_activate=self.threshold_func)
            if y == test_exp_y:
                correct += 1
            else:
                invalid += 1
        
        print(f"Total: {correct}/{correct+invalid} ({(correct/(invalid+correct) * 100):.2f}%)")
                
    
    def image_inspect(self, image_path: str):
        x = get_x_array_by_path_file(image_path)
        y = self.calc_y(x, func_activate=self.threshold_func)
        obj_type = ObjType(y)
        print(f"Preceptron guess it is a: {obj_type.name}!")
        