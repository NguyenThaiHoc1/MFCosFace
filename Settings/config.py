from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_DATASET_FOLDER = BASE_DIR / 'Dataset'

TRAIN_DATASET_RAW_FOLDER = BASE_DATASET_FOLDER / 'raw' / 'lfw'

# ####################################### PREPROCESSING #####################################

TRAIN_DATASET_RESULT_FOLDER = BASE_DATASET_FOLDER / 'result'

# ############################################################################################

TRAIN_DATASET = BASE_DIR / 'Dataset' / 'raw' / 'lfw.tfrecords'

TEST_DATASET = BASE_DIR / 'Dataset' / 'raw' / 'lfw.tfrecords'

CHECKPOINT_SAVE = BASE_DIR / 'Checkpoint'

LOGS_SAVE = BASE_DIR / 'Logs'

MODEL_TYPE = 'NormHead'

NUM_CLASSES = 5749

NUM_IMAGES = 13233

EMBEDDING_SIZE = 512

BATCH_SIZE = 128

MAX_EPOCHS = 100

LEARNING_RATE = 1e-3
