import math
from enum import IntEnum

import numpy.linalg as npla

from Pluggins.face_process.mathlib.umeyama import umeyama
from ProcessingFace.CoreProcess.utlis import *

landmarks_2D_new = np.array([
    [0.000213256, 0.106454],  # 17
    [0.0752622, 0.038915],  # 18
    [0.18113, 0.0187482],  # 19
    [0.29077, 0.0344891],  # 20
    [0.393397, 0.0773906],  # 21
    [0.586856, 0.0773906],  # 22
    [0.689483, 0.0344891],  # 23
    [0.799124, 0.0187482],  # 24
    [0.904991, 0.038915],  # 25
    [0.98004, 0.106454],  # 26
    [0.490127, 0.203352],  # 27
    [0.490127, 0.307009],  # 28
    [0.490127, 0.409805],  # 29
    [0.490127, 0.515625],  # 30
    [0.36688, 0.587326],  # 31
    [0.426036, 0.609345],  # 32
    [0.490127, 0.628106],  # 33
    [0.554217, 0.609345],  # 34
    [0.613373, 0.587326],  # 35
    [0.121737, 0.216423],  # 36
    [0.187122, 0.178758],  # 37
    [0.265825, 0.179852],  # 38
    [0.334606, 0.231733],  # 39
    [0.260918, 0.245099],  # 40
    [0.182743, 0.244077],  # 41
    [0.645647, 0.231733],  # 42
    [0.714428, 0.179852],  # 43
    [0.793132, 0.178758],  # 44
    [0.858516, 0.216423],  # 45
    [0.79751, 0.244077],  # 46
    [0.719335, 0.245099],  # 47
    [0.254149, 0.780233],  # 48
    [0.726104, 0.780233],  # 54
], dtype=np.float32)


class FaceType(IntEnum):
    # enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20
    MARK_ONLY = 100,  # no align at all, just embedded faceinfo


FaceType_to_padding_remove_align = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.40, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}


#  def get_transform_mat_V2(image_landmarks, output_size, face_type, scale=1.0):
class FaceAligment(object):
    @staticmethod
    def active(landmark, output_size, face_type, scale=1.0):
        centered_face = landmark - landmark.mean(axis=0)
        if not isinstance(landmark, np.ndarray):
            image_landmarks = np.array(landmark)

        mat = umeyama(np.concatenate([landmark[17:49], landmark[54:55]]), landmarks_2D_new, True)[0:2]

        g_p = transform_points(np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]), mat, True)
        g_c = g_p[4]

        tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
        tb_diag_vec /= npla.norm(tb_diag_vec)
        bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
        bt_diag_vec /= npla.norm(bt_diag_vec)

        padding, remove_align = FaceType_to_padding_remove_align.get(face_type, 0.0)
        mod = (1.0 / scale) * (npla.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))

        if face_type == FaceType.WHOLE_FACE:
            # adjust vertical offset for WHOLE_FACE, 7% below in order to cover more forehead
            vec = (g_p[0] - g_p[3]).astype(np.float32)
            vec_len = npla.norm(vec)
            vec /= vec_len
            g_c += vec * vec_len * 0.07

        elif face_type == FaceType.HEAD:
            # assuming image_landmarks are 3D_Landmarks extracted for HEAD,
            # adjust horizontal offset according to estimated yaw
            yaw = estimate_averaged_yaw(transform_points(landmark, mat, False))

            hvec = (g_p[0] - g_p[1]).astype(np.float32)
            hvec_len = npla.norm(hvec)
            hvec /= hvec_len

            yaw *= np.abs(math.tanh(yaw * 2))  # Damp near zero

            g_c -= hvec * (yaw * hvec_len / 2.0)

            # adjust vertical offset for HEAD, 50% below
            vvec = (g_p[0] - g_p[3]).astype(np.float32)
            vvec_len = npla.norm(vvec)
            vvec /= vvec_len
            g_c += vvec * vvec_len * 0.50

        if not remove_align:
            l_t = np.float32([g_c - tb_diag_vec * mod,
                              g_c + bt_diag_vec * mod,
                              g_c + tb_diag_vec * mod])
        else:
            l_t = np.array([g_c - tb_diag_vec * mod,
                            g_c + bt_diag_vec * mod,
                            g_c + tb_diag_vec * mod,
                            g_c - bt_diag_vec * mod,
                            ])

            # get area of face square in global space
            area = polygon_area(l_t[:, 0], l_t[:, 1])

            # calc side of square
            side = np.float32(math.sqrt(area) / 2)

            # calc 3 points with unrotated square
            l_t = np.float32([g_c + [-side, -side],
                              g_c + [side, -side],
                              g_c + [side, side]])

        # calc affine transform from 3 global space points to 3 local space points size of 'output_size'
        pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
        mat = cv2.getAffineTransform(l_t, pts2)
        return mat
