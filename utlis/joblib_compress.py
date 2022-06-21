import joblib


class DataSerializer(object):

    def __init__(self):
        self.np_images = None
        self.labels = None

    def set_info(self, np_images, labels):
        print("Set data serializer successfully.")
        self.np_images = np_images
        self.labels = labels

    def get_info(self):
        return self.np_images, self.labels

    def compress(self, path_name, type_compess='dict'):
        assert self.np_images is not None, "Please set value at set_info function..."
        compress_info = None
        if type_compess == 'dict':
            compress_info = dict({
                'list_images': self.np_images,
                'list_labels': self.labels
            })

        if compress_info is None:
            raise ValueError("Error in dump file PKL Please check.")

        joblib.dump(value=compress_info, filename=path_name)
        print("Compress file is Done.")

    def load(self, path_name, type_compess='dict'):
        data = joblib.load(path_name)
        if type_compess == 'dict':
            self.np_images = data['list_images']
            self.labels = data['list_labels']
        else:
            raise NotImplementedError

        print("Loading data and label is successfully.")
