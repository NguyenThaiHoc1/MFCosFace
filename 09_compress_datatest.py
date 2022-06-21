import cv2
import joblib
import argparse
import numpy as np
from Settings import config
from utlis.joblib_compress import DataSerializer
from utlis.evalute import read_pairs, get_paths


def preprocess(np_array):
    img = cv2.resize(np_array, (160, 160))
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


def args_parser():
    parser = argparse.ArgumentParser("Testing function evalute for training.")
    parser.add_argument('--pair_file', required=False,
                        default=r'./Dataset/pair/mfr2_pairs.txt',
                        help='file pairs test')
    parser.add_argument('--folder_data', required=False,
                        default=r'./Dataset/raw/mfr2')
    parser.add_argument('--folder_save_np', required=False,
                        default=r'./Dataset/compress_test/mfr2/test_np_array_mfr2.pkl')
    parser.add_argument('--folder_save_label', required=False,
                        default=r'./Dataset/compress_test/mfr2/test_label_mfr2.pkl')
    return parser.parse_args()


def read_file(np_file, label_file):
    np_images = joblib.load(np_file)
    label_images = joblib.load(label_file)

    return np_images, label_images


if __name__ == '__main__':
    # load PKL file from joblib
    # serializer = DataSerializer()
    # serializer.load(path_name=config.PATH_TEST_PKL)
    # a, b = serializer.get_info()
    # print(a.shape)
    # print(b)
    # print(len(b))

    # create PKL file from joblib
    args = args_parser()

    # loading read_pairs image
    pairs = read_pairs(pairs_filename=args.pair_file)

    paths, actual_issame = get_paths(args.folder_data, pairs)

    list_array = pipline(paths)

    # compress data-test
    serializer = DataSerializer()
    serializer.set_info(np_images=list_array, labels=actual_issame)
    serializer.compress(path_name=config.PATH_TEST_PKL)
