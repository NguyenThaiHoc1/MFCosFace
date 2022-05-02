"""
    Author: Nguyen Thai Hoc
    Date: 29/04/2022
    This file is Reprocess which create new dataset with feature like:
        - Detection
        - Augmentation
        - Alignment

"""
import logging
import os
import shutil
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from ProcessingFace.Pipeline.pipelineface import PipelineFace
from ProcessingFace.Pipeline.pipelinemask import PipelineMaskTheFace
from Settings import config
from Tensorflow.TFRecord import tfrecord
from utlis.argsparse import parser_generator_mask

log_path = Path('./log.txt')

logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

file = open(log_path, "w")
file.close()


class ImageClass(object):
    def __init__(self, filepath):
        self.img_path = filepath
        self.np_image = self.read()

    def read(self):
        image = cv2.imread(self.img_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def setbbox(self, bbox):
        self.bbox = bbox

    def setlandmarks(self, landmarks):
        self.landmarks = landmarks

    def info(self):
        assert self.np_image is not None, "Pls, init numpy image ..."
        print(f"Shape of image: {self.np_image.shape}")

    def show(self):
        assert self.np_image is not None, "Pls, init numpy image ..."
        plt.figure(figsize=(5, 5))
        plt.imshow(self.np_image)
        plt.axis('off')

    def showface(self):
        assert self.bbox is not None, "Pls, updated bbox-face of image ..."
        clone_np_img = self.np_image.copy()
        xmin, ymin, xmax, ymax = self.bbox
        clone_np_img = cv2.rectangle(clone_np_img, (xmin, ymin), (xmax, ymax), (0, 155, 255), 2)
        plt.figure(figsize=(5, 5))
        plt.imshow(clone_np_img)
        plt.axis('off')

    def showlandmarks(self):
        assert self.landmarks is not None, "Pls, updated landmarks-face of image ..."
        img = self.np_image.copy()
        for idx, landmark in enumerate(self.landmarks):
            for point in landmark:
                pos = (point[0], point[1])
                cv2.circle(img, pos, 1, color=(0, 255, 255))
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')


def test():
    pipeline = PipelineFace(output_size=128)
    image = ImageClass(filepath='./Dataset/raw/lfw/Saddam_Hussein/Saddam_Hussein_0008.jpg')
    face = pipeline(image.np_image)
    plt.imshow(face[0])
    plt.show()

    pipeline_mask = PipelineMaskTheFace()

    list_face_mask, _, _, _ = pipeline_mask.active(face[0], mask_type='surgical')

    plt.imshow(list_face_mask[0])
    plt.show()


def main():
    args = parser_generator_mask()

    path_root_dataset = config.TRAIN_DATASET_RAW_FOLDER

    path_result_dataset = config.TRAIN_DATASET_RESULT_FOLDER / args.name_dataset

    path_result_align = path_result_dataset / 'align'

    path_result_mask = path_result_dataset / 'mask'

    if path_result_dataset.exists():
        shutil.rmtree(path=path_result_dataset)
        os.makedirs(path_result_dataset)
    else:
        os.makedirs(path_result_dataset)

    if path_result_align.exists():
        shutil.rmtree(path=path_result_align)
        os.makedirs(path_result_align)
    else:
        os.makedirs(path_result_align)

    if path_result_mask.exists():
        shutil.rmtree(path=path_result_mask)
        os.makedirs(path_result_mask)
    else:
        os.makedirs(path_result_mask)

    pipeline = PipelineFace(output_size=128)

    pipeline_mask = PipelineMaskTheFace()

    # Loading dataset
    dataset = tfrecord.TFRecordData().load(config.TRAIN_DATASET,
                                           is_repeat=False,
                                           binary_img=True,
                                           is_crop=False,
                                           reprocess=True,
                                           shuffle=False,
                                           batch_size=1)

    for data in tqdm(dataset):
        inputs, path_image = data

        np_image = inputs[0].numpy()

        abs_path = bytes.decode(path_image.numpy()[0])

        new_path = Path(abs_path)

        file_name = new_path.stem

        name_folder = str(new_path.parent).split('/')[-1]

        base_filename = 'align_' + file_name + '.jpg'

        align_save_path = path_result_align / name_folder

        new_filename_align = align_save_path / base_filename

        if not align_save_path.exists():
            os.makedirs(align_save_path)

        try:

            face = pipeline(np_image)

            if not args.check_mask:
                break

            list_type_mask = ['cloth', 'surgical', 'surgical_blue']

            abs_path_id_mask = path_result_mask / name_folder

            if not abs_path_id_mask.exists():
                os.makedirs(abs_path_id_mask)

            for type_mask in list_type_mask:
                list_face_mask, _, _, _ = pipeline_mask.active(face[0], mask_type=type_mask)

                base_filename_mask = f'mask-{type_mask}_' + file_name + '.jpg'

                abs_path_mask_filename = abs_path_id_mask / base_filename_mask

                plt.imsave(abs_path_mask_filename, list_face_mask[0])

            plt.imsave(new_filename_align, face[0])

        except Exception as ex:
            logging.info(f"Error: {name_folder} - {abs_path}")
            pass

    logging.info("DONE")


if __name__ == '__main__':
    test()
