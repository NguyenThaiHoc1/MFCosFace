import os
import shutil

from tqdm import tqdm

from Settings import config


def check_folder(path):
    if path.exists():
        shutil.rmtree(path=path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def copy(origin_folder, dist_folder):
    for filename in origin_folder.glob('*.jpg'):
        abs_source = origin_folder / filename
        shutil.copy(abs_source, dist_folder)


def merge():
    dataset_1 = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'mask'

    dataset_2 = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'align'

    dataset_merge = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'merge'

    # check folder
    check_folder(dataset_merge)

    list_name_mask = os.listdir(dataset_1)

    list_name_align = os.listdir(dataset_2)

    for id_name in tqdm(list_name_align):

        source_path_align = dataset_2 / id_name

        source_path_mask = dataset_1 / id_name

        if len(os.listdir(source_path_mask)) == 0 or len(os.listdir(source_path_align)) == 0:
            print(id_name)
            continue

        abs_path_merge = dataset_merge / id_name

        check_folder(abs_path_merge)

        # align
        copy(source_path_align, abs_path_merge)
        copy(source_path_mask, abs_path_merge)


if __name__ == '__main__':
    merge()
