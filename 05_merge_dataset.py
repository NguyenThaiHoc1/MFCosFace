import os
from pathlib import Path
from tqdm import tqdm

from Settings import config
from utlis import utlis


def merge():
    dataset_1 = Path('/Volumes/Ventoy') / 'Data' / 'reprocess' / 'casia_part1' / 'mask'

    dataset_2 = Path('/Volumes/Ventoy') / 'Data' / 'reprocess' / 'casia_part1' / 'align'

    dataset_merge = Path('/Volumes/Ventoy') / 'Data' / 'reprocess' / 'casia_part1' / 'merge'

    # check folder
    utlis.check_folder(dataset_merge)

    list_name_align = os.listdir(dataset_2)

    for id_name in tqdm(list_name_align):

        if id_name[0] == '.':
            continue

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
