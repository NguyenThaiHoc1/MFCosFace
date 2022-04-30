import face_alignment
import numpy as np
import tensorflow as tf


def shape_to_landmarks(shape):
    face_landmarks = dict()
    face_landmarks["left_eyebrow"] = [
        tuple(shape[17]),
        tuple(shape[18]),
        tuple(shape[19]),
        tuple(shape[20]),
        tuple(shape[21]),
    ]
    face_landmarks["right_eyebrow"] = [
        tuple(shape[22]),
        tuple(shape[23]),
        tuple(shape[24]),
        tuple(shape[25]),
        tuple(shape[26]),
    ]
    face_landmarks["nose_bridge"] = [
        tuple(shape[27]),
        tuple(shape[28]),
        tuple(shape[29]),
        tuple(shape[30]),
    ]
    face_landmarks["nose_tip"] = [
        tuple(shape[31]),
        tuple(shape[32]),
        tuple(shape[33]),
        tuple(shape[34]),
        tuple(shape[35]),
    ]
    face_landmarks["left_eye"] = [
        tuple(shape[36]),
        tuple(shape[37]),
        tuple(shape[38]),
        tuple(shape[39]),
        tuple(shape[40]),
        tuple(shape[41]),
    ]
    face_landmarks["right_eye"] = [
        tuple(shape[42]),
        tuple(shape[43]),
        tuple(shape[44]),
        tuple(shape[45]),
        tuple(shape[46]),
        tuple(shape[47]),
    ]
    face_landmarks["top_lip"] = [
        tuple(shape[48]),
        tuple(shape[49]),
        tuple(shape[50]),
        tuple(shape[51]),
        tuple(shape[52]),
        tuple(shape[53]),
        tuple(shape[54]),
        tuple(shape[60]),
        tuple(shape[61]),
        tuple(shape[62]),
        tuple(shape[63]),
        tuple(shape[64]),
    ]
    face_landmarks["bottom_lip"] = [
        tuple(shape[54]),
        tuple(shape[55]),
        tuple(shape[56]),
        tuple(shape[57]),
        tuple(shape[58]),
        tuple(shape[59]),
        tuple(shape[48]),
        tuple(shape[64]),
        tuple(shape[65]),
        tuple(shape[66]),
        tuple(shape[67]),
        tuple(shape[60]),
    ]
    face_landmarks["chin"] = [
        tuple(shape[0]),
        tuple(shape[1]),
        tuple(shape[2]),
        tuple(shape[3]),
        tuple(shape[4]),
        tuple(shape[5]),
        tuple(shape[6]),
        tuple(shape[7]),
        tuple(shape[8]),
        tuple(shape[9]),
        tuple(shape[10]),
        tuple(shape[11]),
        tuple(shape[12]),
        tuple(shape[13]),
        tuple(shape[14]),
        tuple(shape[15]),
        tuple(shape[16]),
    ]
    return face_landmarks


class FaceLandmark(object):
    def __init__(self):
        self.aligner = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            device='cuda' if tf.test.is_gpu_available() else 'cpu',
            face_detector='blazeface', )

    def active(self, image, bbox):
        # clone object
        clone_np_image = image.copy()

        # convert bbox
        np_box = np.array(bbox)
        np_box.astype(int)

        # get landmarks
        points = self.aligner.get_landmarks(clone_np_image, detected_faces=[np_box])
        return points
