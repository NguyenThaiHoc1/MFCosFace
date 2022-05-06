import cv2
import tensorflow as tf

from Settings import config
from Tensorflow.Architecture.ModelFeatureExtraction.inception_resnet_v1 import InceptionResNetV1
from utlis.argsparse import parser_test
from utlis.evalute import *


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
    # PARAMETER
    args = parser_test()

    # DATA
    pairs = read_pairs(pairs_filename=args.file_pair)

    paths, actual_issame = get_paths(args.folder_data, pairs)

    list_array = pipline(paths)

    # MODEL
    model = InceptionResNetV1(embedding_size=config.EMBEDDING_SIZE, name="InceptionResNetV1")

    # Loading checkpoint (if you have)
    checkpoint_path = tf.train.latest_checkpoint(config.CHECKPOINT_SAVE)
    print('[*] load ckpt from {}.'.format(checkpoint_path))
    model.load_weights(checkpoint_path)

    # EVALUATE
    distances, labels = evalute(embedding_size=config.EMBEDDING_SIZE,
                                batch_size=config.BATCH_SIZE,
                                model=model,
                                carray=list_array, issame=actual_issame)
