import os
from pathlib import Path

from Settings import config
from Tensorflow.TFRecord import tfrecord

if __name__ == '__main__':

    path_check = True

    if path_check:

        dataset = Path(
            '/Volumes/Ventoy') / 'Data' / 'data_fujinet_v2'  # 'data_fujinet_v2'  # config.TRAIN_DATASET_RAW_FOLDER  #  # config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'merge'

        extentions_img = '*.jpg'
        dict_count = {
            'num_classes': 0,
            'num_images': 0
        }
        for folder_name in os.listdir(dataset):

            folder_path = dataset / str(folder_name)

            len_file = os.listdir(folder_path)

            if len(len_file) == 0:
                print(folder_name)

            for filename in folder_path.glob(extentions_img):
                dict_count['num_images'] += 1

            dict_count['num_classes'] += 1

        print(f"Number of Person: {dict_count['num_classes']} - Number of images: {dict_count['num_images']}")

    else:

        dataset = config.TRAIN_DATASET_RESULT_FOLDER / 'data_cfp_lfw.tfrecords'

        loader_dataset = tfrecord.TFRecordData().load(dataset,
                                                      is_repeat=False,
                                                      binary_img=True,
                                                      is_crop=False,
                                                      reprocess=False,
                                                      shuffle=False,
                                                      batch_size=1)
        num_classes = set()
        num_images = 0
        for inputs, labels in loader_dataset:

            num_classes.add(int(labels.numpy()[0]))
            num_images += 1

        print(f"Number of Person: {len(num_classes)} - Number of images: {num_images}")



