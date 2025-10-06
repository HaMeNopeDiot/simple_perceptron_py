import kagglehub
import random
import os

from simple_preceptron  import SimplePreceptron
from conversion import get_x_array_by_path_file
from utility    import TRAIN_EXAMPLES, OBJ2TRAIN_FST, OBJ2TRAIN_SND,\
                        OBJ2TRAIN_FST_VAL, OBJ2TRAIN_SND_VAL, DEFAULT_DATASET_DIR, DEFAULT_OBJ_TRAIN_DATASET_PATH,\
                            PROGRAM_MODE, W_FILE_NAME, ProgramMode

from utility    import TESTING_ENABLE, TEST_ONE_PICTURE, IMAGE_PATH

# ============================================================
# Some helpers
# ============================================================

def download_dataset():
    # Download latest version
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)
    os.environ['KAGGLEHUB_CACHE'] = './'
    path = kagglehub.dataset_download("reevald/geometric-shapes-mathematics")
    print("Path to dataset files:", path)

    
def prepare_train_set():
    train_set = []
    print(f"Start to get train sets")
    for i in range(TRAIN_EXAMPLES):
        # Get data
        train_obj1_x = get_x_array_by_path_file(f"{DEFAULT_OBJ_TRAIN_DATASET_PATH}/train/{OBJ2TRAIN_FST}/{OBJ2TRAIN_FST}-{i}.jpg")
        train_obj2_x = get_x_array_by_path_file(f"{DEFAULT_OBJ_TRAIN_DATASET_PATH}/train/{OBJ2TRAIN_SND}/{OBJ2TRAIN_SND}-{i}.jpg")   
        # Make a list of sets with two values: [X_values, Y_expected]
        train_set.append([train_obj1_x, OBJ2TRAIN_FST_VAL]) 
        train_set.append([train_obj2_x, OBJ2TRAIN_SND_VAL])
        if i % 100 == 0:
            print(f"{i} case passed. ({((i/TRAIN_EXAMPLES)*100):.2f}%)")
    # Shuffle values between us
    random.shuffle(train_set)
    return train_set
        
    
# ============================================================
# MAIN CODE
# ============================================================

if not os.path.isdir("./" + DEFAULT_DATASET_DIR):
    download_dataset()
    
simple_preceptron = SimplePreceptron(cnt_cells=224 * 224, learning_rate=0.7, max_epochs=1400)


if PROGRAM_MODE == ProgramMode.FIT:
    # FIT MODEL
    print(f"Start to fit model")
    train_set = prepare_train_set()
    simple_preceptron.fill_weigths()
    simple_preceptron.fit(train_set, name_export=W_FILE_NAME)
elif PROGRAM_MODE == ProgramMode.EXPORT:
    # FOR ALREADY FITTED MODEL
    print(f"Import model...")  
    simple_preceptron.import_weigths(W_FILE_NAME)
else:
    assert False, "Unknown FIT_MODEL mode..."

# TEST PRECEPTRON
if TESTING_ENABLE:
    print(f"Testing phase started...")
    simple_preceptron.test(300)
    
if TEST_ONE_PICTURE:
    simple_preceptron.image_inspect(IMAGE_PATH)