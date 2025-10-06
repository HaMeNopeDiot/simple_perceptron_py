from enum import Enum

# DEFINE TYPES OF OBJECTS
OBJ2TRAIN_FST       = "rectangle"
OBJ2TRAIN_SND       = "triangle"

OBJ2TRAIN_FST_VAL   = 0
OBJ2TRAIN_SND_VAL   = 1

class ObjType(Enum):
    RECTANGLE = 0
    TRIANGLE  = 1

class ProgramMode(Enum):
    EXPORT  = 0
    FIT     = 1

# TESTS COUNT
TRAIN_EXAMPLES = 1500
TEST_EXAMPLES = 500
TEST_OFFSET = 2000


# PATH GLOBAL VARS
DEFAULT_W_DIR = "weigths"
DEFAULT_DATASET_DIR = "datasets"
DEFAULT_OBJ_TRAIN_DATASET_PATH = "datasets/reevald/geometric-shapes-mathematics/versions/4/dataset"

# PARAMETERS
PROGRAM_MODE = ProgramMode.EXPORT
W_FILE_NAME = "weigth-5"

TESTING_ENABLE      = False
TEST_ONE_PICTURE    = True

IMAGE_PATH = "./my_image.jpg"