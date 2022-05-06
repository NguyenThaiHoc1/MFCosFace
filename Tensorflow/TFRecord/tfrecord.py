import glob
import logging
import os
import random

import tensorflow as tf
from tqdm import tqdm

from Tensorflow.TFParserRecord.parse_record import ParserRecord
from Tensorflow.TFRecord.tfsample import TFSample


class TFRecordData(object):

    num_class_count = 0
    samples = []

    @staticmethod
    def active(path_output_record):
        logging.info(f"Writing TFRecord file ...")
        with tf.io.TFRecordWriter(path=path_output_record) as writer:
            for img_path, real_name, filename, id_name in tqdm(TFRecordData.samples):
                sample = TFSample.create(np_image=open(img_path, 'rb').read(),
                                         id_name=int(id_name),
                                         filename=str.encode(filename),
                                         img_path=str.encode(img_path),
                                         real_name=str.encode(real_name))
                writer.write(record=sample.SerializeToString())
        logging.info(f"Writing TFRecord file done.")

    @staticmethod
    def create(path_dataset):
        """
        Create TFRecord file
        :return:
        """
        if not os.path.isdir(path_dataset):
            logging.error("Please check path dataset is not a folder.")
        else:
            logging.info(f"Creating TFRecord file from: {path_dataset} ... ")

        logging.info(f"Reading data list ...")
        for index_name, real_name in tqdm(enumerate(os.listdir(path_dataset))):
            list_id_filename = glob.glob(os.path.join(path_dataset, real_name, "*.jpg"))
            for img_path in list_id_filename:
                filename = os.path.join(real_name, os.path.basename(img_path))
                TFRecordData.samples.append((img_path, real_name, filename, TFRecordData.num_class_count))

            TFRecordData.num_class_count += 1
        random.shuffle(TFRecordData.samples)

        logging.info(f"Reading data list done.")

        logging.info(f"Creating TFRecord file done.")

    @staticmethod
    def load(record_name, shuffle, batch_size,
             is_repeat=True,
             binary_img=False, is_crop=False,
             reprocess=True,
             buffer_size=10240):
        """
        Load TFRecord file
        :return:
        """
        """load dataset from TFRecord"""
        logging.info(f"Loading TFRecord file ...")
        raw_dataset = tf.data.TFRecordDataset(record_name)

        if is_repeat:
            raw_dataset = raw_dataset.repeat()

        if shuffle:
            raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)

        if reprocess:
            dataset = raw_dataset.map(
                ParserRecord(binary_img=binary_img, is_crop=is_crop,
                             process=reprocess),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        else:

            dataset = raw_dataset.map(
                ParserRecord(binary_img=binary_img, is_crop=is_crop),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        logging.info(f"Loading TFRecord done.")
        return dataset
