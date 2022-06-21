from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_DATASET_FOLDER = BASE_DIR / 'Dataset'

TRAIN_DATASET_RAW_FOLDER = BASE_DATASET_FOLDER / 'raw' / 'lfw'

# ####################################### PREPROCESSING #####################################

TRAIN_DATASET_RESULT_FOLDER = BASE_DATASET_FOLDER / 'result'

# ############################################################################################

TRAIN_DATASET = BASE_DIR / 'Dataset' / 'result' / 'data_cfp_lfw.tfrecords'

VAL_DATASET = BASE_DIR / 'Dataset' / 'result' / 'data_cfp_lfw.tfrecords'

CHECKPOINT_SAVED = BASE_DIR / 'Checkpoint'

CHECKPOINT_NEW_SAVE = BASE_DIR / 'NEW_Checkpoint'

TENSORBOARD_SAVE = BASE_DIR / 'Visualization'

LOGS_SAVE = BASE_DIR / 'Logs' / 'logs.txt'

MODEL_TYPE = 'ArcHead'  # NormHead

NUM_CLASSES = 6239

NUM_IMAGES = 72768

EMBEDDING_SIZE = 512

BATCH_SIZE = 128

MAX_EPOCHS = 1000

LEARNING_RATE = 1e-3

#########################

DLIB_LANDMARK_MODEL = BASE_DIR / 'Model' / 'shape_predictor_68_face_landmarks.dat'

######################## TEST

PATH_TEST_PKL = BASE_DIR / 'Dataset' / 'compress_test' / 'test_mfr2.pkl'
