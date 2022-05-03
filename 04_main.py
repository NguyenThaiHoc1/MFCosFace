import logging

from Settings import config
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from Tensorflow.TFloader import DataLoader
from trainer import Trainer
from utlis.utlis import *


def create_log_summary(path):
    logging.basicConfig(filename=path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    file = open(path, "w")
    file.close()


def check_all_folder():
    if not config.CHECKPOINT_SAVE.exists():
        os.makedirs(config.CHECKPOINT_SAVE)


def main():
    # init
    create_log_summary('log.txt')
    check_all_folder()

    # Data
    logging.info("* STEP 1: Creating loader ...")
    loader = DataLoader(path_train=config.TRAIN_DATASET,
                        path_test=config.TEST_DATASET,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        binary_status=True,
                        augmentation=True)
    logging.info("* STEP 1: Creating loader done.")

    # Model
    logging.info("* STEP 2: Creating Architecture ...")
    model = InceptionResNetV1(num_classes=config.NUM_CLASSES,
                              embedding_size=config.EMBEDDING_SIZE,
                              model_type=config.MODEL_TYPE,
                              name="InceptionResNetV1")
    logging.info("* STEP 2: Creating Architecture done.")

    # Loading checkpoint (if you have)
    logging.info("* STEP 3: Loading checkpoint ...")
    current_epochs, steps = load_checkpoint(path_checkpoint=config.CHECKPOINT_SAVE,
                                            model=model, steps_per_epoch=loader.steps_per_epoch_train)
    logging.info("* STEP 3: Loading checkpoint done.")

    # Training
    logging.info("* STEP 4: Loading trainer ...")
    face_trainer = Trainer(loader=loader,
                           model=model,
                           steps=steps,
                           current_epochs=current_epochs,
                           max_epochs=config.MAX_EPOCHS,
                           learning_rate=config.LEARNING_RATE,
                           tensorboard_path=config.TENSORBOARD_SAVE,
                           save_path=config.CHECKPOINT_SAVE,
                           logs=config.LOGS_SAVE,
                           loss_type='Softmax')

    logging.info("* STEP 4: Loading trainer done.")

    logging.info("* STEP 5: Training ...")
    face_trainer.training()
    logging.info("* STEP 5: Training done.")


if __name__ == '__main__':
    main()
