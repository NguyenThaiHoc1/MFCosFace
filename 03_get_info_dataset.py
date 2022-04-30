import os

from Settings import config

if __name__ == '__main__':

    dataset = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'mask'
    extentions_img = '*.jpg'
    dict_count = {
        'num_classes': 0,
        'num_images': 0
    }
    for folder_name in os.listdir(dataset):

        folder_path = dataset / str(folder_name)

        for filename in folder_path.glob(extentions_img):
            dict_count['num_images'] += 1

        dict_count['num_classes'] += 1

    print(f"Number of Person: {dict_count['num_classes']} - Number of images: {dict_count['num_images']}")
