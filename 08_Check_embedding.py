import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from Settings import config
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from utlis.utlis import l2_norm


def loading_model():
    # MODEL
    model = InceptionResNetV1(embedding_size=config.EMBEDDING_SIZE, name="InceptionResNetV1")

    # Loading checkpoint (if you have)
    path_checkpoint = Path('/Volumes/Ventoy/Data/Checkpoint/NEW_Checkpoint')
    checkpoint_path = tf.train.latest_checkpoint(path_checkpoint)
    print('[*] load ckpt from {}.'.format(checkpoint_path))
    model.load_weights(checkpoint_path)
    return model


def preprocess(np_array):
    img = cv2.resize(np_array, (112, 112))
    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def pipline(path_array):
    np_images = []
    for abspath_filename in path_array:
        np_image = cv2.imread(abspath_filename)
        np_preprocessed = preprocess(np_image)
        np_images.append(np_preprocessed)
    return np.asarray(np_images)


if __name__ == '__main__':
    folder_name_check = 'hoc-nt-mask_v3'

    path_register = Path('/Volumes/Ventoy/Data/face_register') / folder_name_check

    path_checked = Path('/Volumes/Ventoy/Data/face_register') # / 'hoc-nt-mask_v1'

    # path_checked = Path('/Volumes/Ventoy/Data/data_fujinet')

    model = loading_model()

    for path_re in os.listdir(path_register):

        if path_re[0] == '.':
            continue

        abs_path = path_register / path_re
        np_reprocessed_face = pipline([str(abs_path)])

        for person_id in os.listdir(path_checked):

            if person_id[0] == '.':
                continue

            if person_id == folder_name_check:
                continue

            # if not person_id == 'khanh-td':
            #     continue

            list_numpy_compare = []
            path_person_id = path_checked / person_id

            for path_check in os.listdir(path_person_id):
                if path_check[0] == '.':
                    continue
                abs_path_regis = path_person_id / path_check
                list_numpy_compare.append(str(abs_path_regis))

            list_face_registered = pipline(list_numpy_compare)

            np_embedding_face = l2_norm(model(np_reprocessed_face, training=False))
            list_embedding_registered = l2_norm(model(list_face_registered, training=False))

            diff = np.subtract(np_embedding_face, list_embedding_registered)
            dist = np.sum(np.square(diff), axis=1)

            mean_rec = np.mean(dist)

            if mean_rec < 1.4:
                print(f"{person_id} -- {mean_rec}")
