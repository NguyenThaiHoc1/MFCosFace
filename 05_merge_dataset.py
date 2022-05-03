import os

from tqdm import tqdm

from Settings import config
from utlis import utlis


def merge():
    dataset_1 = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'mask'

    dataset_2 = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'align'

    dataset_merge = config.TRAIN_DATASET_RESULT_FOLDER / 'lfw' / 'merge'

    # check folder
    utlis.check_folder(dataset_merge)

    list_name_align = os.listdir(dataset_2)

    for id_name in tqdm(list_name_align):

        source_path_align = dataset_2 / id_name

        source_path_mask = dataset_1 / id_name

        if len(os.listdir(source_path_mask)) == 0 or len(os.listdir(source_path_align)) == 0:
            print(id_name)
            continue

        abs_path_merge = dataset_merge / id_name

        utlis.check_folder(abs_path_merge)

        # align
        utlis.copy(source_path_align, abs_path_merge)
        utlis.copy(source_path_mask, abs_path_merge)


if __name__ == '__main__':
    merge()
