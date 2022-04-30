from Tensorflow.TFRecord.tfrecord import TFRecordData


class DataLoader(object):

    def __init__(self, path_train, path_test, batch_size, binary_status, shuffle, augmentation):
        """
        DataLoader is the class which help to make a TFDataset
        :param path_train: path to TFRecord file
        :param path_test:  path to TFRecord file
        :param batch_size: amount of the sample via network
        :param binary_status: mode read data of TFRecord
        :param shuffle: shuffle data while loop in data
        :param augmentation: process data more various data
        """
        self.path_train = path_train
        self.path_test = path_test

        self.batch_size = batch_size
        self.binary_status = binary_status
        self.shuffle = shuffle
        self.augmentation = augmentation

        # samples train - test
        self.num_samples_train = self._get_total_samples_train()
        self.num_samples_test = self._get_total_samples_test()

    def _get_total_samples_train(self):
        loader = TFRecordData().load(self.path_train, binary_img=self.binary_status,
                                     shuffle=False, is_repeat=False, batch_size=1)
        count = sum([1 for idx in loader])
        return count

    def _get_total_samples_test(self):
        loader = TFRecordData().load(self.path_test, binary_img=self.binary_status,
                                     shuffle=False, is_repeat=False, batch_size=1)
        count = sum([1 for idx in loader])
        return count

    @property
    def train(self):
        return TFRecordData().load(self.path_train, binary_img=self.binary_status,
                                   is_crop=self.augmentation, shuffle=self.shuffle, batch_size=self.batch_size)

    @property
    def test(self):
        return TFRecordData().load(self.path_test, binary_img=self.binary_status,
                                   is_crop=self.augmentation, shuffle=self.shuffle, batch_size=self.batch_size)

    @property
    def steps_per_epoch_train(self):
        dataset_len = self.num_samples_train
        steps_per_epoch = dataset_len // self.batch_size
        return steps_per_epoch
