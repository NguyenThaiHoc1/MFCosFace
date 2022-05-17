import argparse


def parser_record():
    parser = argparse.ArgumentParser(description='Parameter for create TFRecord.')
    parser.add_argument('--path', required=True, nargs='+',
                        help='Where contains dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Where file tfrecord')

    return parser.parse_args()


def parser_generator_mask():
    parser = argparse.ArgumentParser(description='Parameter for generator Mask.')
    parser.add_argument('--name_dataset', type=str, required=True, help='name of dataset')
    parser.add_argument('--check_mask', type=bool, required=True,
                        help='Where variable process mask for person')

    return parser.parse_args()


def parser_test():
    parser = argparse.ArgumentParser(description='Parameter for generator Mask.')
    parser.add_argument('--file_pair', type=str, required=True, help='path file pair')
    parser.add_argument('--folder_data', type=str, required=True, help='folder which contains image from pair file')
    return parser.parse_args()


def parser_summary_data_fujinet_each_person():
    parser = argparse.ArgumentParser(description='Parameter for generator Mask.')
    parser.add_argument('--folder_source_data_path', type=str, required=True,
                        help='folder which contains all data raw')
    parser.add_argument('--folder_dist_data_path', type=str, required=True,
                        help='where contains all data')
    return parser.parse_args()
