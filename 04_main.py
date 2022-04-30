import logging

from Settings import config
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from Tensorflow.TFloader import DataLoader
from trainer import Trainer
from utlis.utlis import *

logging.basicConfig(level=logging.INFO)


def main():
    # Data
    logging.info("Creating loader ...")
    loader = DataLoader(path_train=config.TRAIN_DATASET,
                        path_test=config.TEST_DATASET,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        binary_status=True,
                        augmentation=True)
    logging.info("Creating loader done.")

    # Model
    logging.info("Creating Architecture ...")
    model = InceptionResNetV1(num_classes=config.NUM_CLASSES,
                              embedding_size=config.EMBEDDING_SIZE,
                              model_type=config.MODEL_TYPE,
                              name="InceptionResNetV1")
    logging.info("Creating Architecture done.")

    # Loading checkpoint (if you have)
    logging.info("Loading checkpoint ...")
    current_epochs, steps = load_checkpoint(path_checkpoint=config.CHECKPOINT_SAVE,
                                            model=model, steps_per_epoch=loader.steps_per_epoch_train)
    logging.info("Loading checkpoint done.")

    # Training
    logging.info("Loading trainer ...")
    face_trainer = Trainer(loader=loader,
                           model=model,
                           steps=steps,
                           current_epochs=current_epochs,
                           max_epochs=config.MAX_EPOCHS,
                           learning_rate=config.LEARNING_RATE,
                           save_path=config.CHECKPOINT_SAVE,
                           logs=config.LOGS_SAVE,
                           loss_type='Softmax')
    logging.info("Loading trainer done.")

    logging.info("Training ...")
    face_trainer.training()
    logging.info("Training done.")


if __name__ == '__main__':
    main()
