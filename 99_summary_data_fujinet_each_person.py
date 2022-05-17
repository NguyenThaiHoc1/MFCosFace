import os
import shutil
from collections import defaultdict
from os.path import isfile

from tqdm import tqdm

from utlis.argsparse import parser_summary_data_fujinet_each_person


def get_person(path, result):
    for each_folder in tqdm(os.listdir(path)):

        abs_path = os.path.join(path, each_folder)

        if isfile(abs_path):
            name_folder = path.split('/')[7]
            if not name_folder == 'unknown':
                result[str(name_folder)].append(abs_path)

        else:
            get_person(abs_path, result)


def copy_file(result, path_out):
    for key, values in result.items():

        abs_path_person = os.path.join(path_out, key)

        if not os.path.exists(abs_path_person):
            os.makedirs(abs_path_person)

        for value in values:
            shutil.copy(value, abs_path_person)


if __name__ == '__main__':
    args = parser_summary_data_fujinet_each_person()

    dict_information = defaultdict(list)

    get_person(args.folder_source_data_path, dict_information)

    copy_file(dict_information, args.folder_dist_data_path)
