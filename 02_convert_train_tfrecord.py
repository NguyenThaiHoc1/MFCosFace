"""
    Author: Nguyen Thai Hoc
    Date: 19-04-2022
    Purpose: Class TFRecord
"""
import logging
import os
import glob
from Tensorflow.TFRecord import tfrecord
from utlis.argsparse import parser_record

logging.basicConfig(level=logging.INFO)


def _training_step(iter_train):
    inputs, labels = next(iter_train)
    print("=============")


if __name__ == '__main__':
    logging.info("Convert_train_tfrecord.py ...")
    args = parser_record()

    # Create TFRecord
    for file_path in args.path:
        tfrecord.TFRecordData().create(path_dataset=file_path)

    tfrecord.TFRecordData().active(path_output_record=args.output_dir)

    # Load TFRecord
    # dataset = tfrecord.TFRecordData().load(args.output_dir, binary_img=True, is_crop=False,
    #                                        shuffle=True, batch_size=32)

    # for idx in dataset:
    #     print(idx)

    # iter_train = iter(dataset)
    # count = 0
    # while True:
    #     _training_step(iter_train)
    #     count += 1
    #     print(count)

    # root = os.path.abspath(os.path.dirname(__file__))
    # dataset_path = os.path.join(root, "Dataset", "total", "mfr2")
    # count = 0
    # for foldername in os.listdir(dataset_path):
    #     folder_path = os.path.join(dataset_path, foldername)
    #
    #     if not os.path.isdir(folder_path):
    #         continue
    #
    #     for filename in glob.glob(os.path.join(folder_path, '*.png')):
    #         print(filename)
    #         count += 1
    #
    # print(count)
    # ds_size = sum(1 for _ in dataset)

    # print(ds_size)
    # dataset_iter = iter(dataset)
    #
    # for idx in tqdm(range(10)):
    #     _training_step(dataset_iter)
    #
    # logging.info("Conver_train_tfrecord.py done.")
