import tensorflow as tf


# Process image data augmentation
def transform_images_setup(is_crop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (160, 160))
        # x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train

    return transform_images


# process labels
def _transform_targets(y_train):
    return y_train
