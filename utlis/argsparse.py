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
