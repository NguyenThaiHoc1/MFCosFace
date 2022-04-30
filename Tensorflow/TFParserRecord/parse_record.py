import tensorflow as tf

from Tensorflow.TFProcessImage import process_image


class ParserRecord(object):

    def __init__(self, binary_img, is_crop, process=False):
        self.binary_img = binary_img

        self.is_crop = is_crop

        self.reprocess = process

    def __call__(self, record):
        if self.binary_img:
            features = {'image/id_name': tf.io.FixedLenFeature([], tf.int64),
                        'image/np_image': tf.io.FixedLenFeature([], tf.string),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}

            features_record = tf.io.parse_single_example(record, features)
            np_array = tf.image.decode_jpeg(features_record['image/np_image'], channels=3)
        else:
            features = {'image/id_name': tf.io.FixedLenFeature([], tf.string),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}

            features_record = tf.io.parse_single_example(record, features)
            image_encoded = tf.io.read_file(features_record['image/img_path'])
            np_array = tf.image.decode_jpeg(image_encoded, channels=3)

        labels = tf.cast(features_record['image/id_name'], tf.float32)

        if self.is_crop:
            """
                We can build data preprocess in here 
            """
            np_array = process_image.transform_images_setup(is_crop=self.is_crop)(np_array)

        """
            We can build reprocess labels in here 
        """
        # labels = _transform_targets(labels)

        if self.reprocess:
            image_path = tf.cast(features_record['image/img_path'], tf.string)
            return np_array, image_path

        return np_array, labels
