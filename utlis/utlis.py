import glob
import os
import shutil

import numpy as np
import tensorflow as tf


def save_weight(model, path_dir):
    print('[*] save ckpt file!')
    model.save_weights(path_dir)


def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, steps + 1


def load_checkpoint(path_checkpoint, model, steps_per_epoch):
    checkpoint_path = tf.train.latest_checkpoint(path_checkpoint)
    if checkpoint_path is not None:
        print('[*] load ckpt from {}.'.format(checkpoint_path))
        model.load_weights(checkpoint_path)
        epochs, steps = get_ckpt_inf(ckpt_path=checkpoint_path, steps_per_epoch=steps_per_epoch)
    else:
        print('[*] training from scratch.')
        epochs, steps = 1, 1
    return epochs, steps


def get_num_samples(path):
    count = 0
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for filename in glob.glob(os.path.join(folder_path, '*.png')):
            print(filename)
            count += 1
    return count


def check_folder(path):
    if path.exists():
        shutil.rmtree(path=path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def copy(origin_folder, dist_folder):
    for filename in origin_folder.glob('*.jpg'):
        check_store = str(filename.stem)[0]
        if check_store == '.':
            continue
        abs_source = origin_folder / filename
        shutil.copy(abs_source, dist_folder)


def l2_norm(embedings, axis=1):
    norm = np.linalg.norm(embedings, axis=axis, keepdims=True)
    output = embedings / norm
    return output
