import cv2
from ProcessingFace.CoreProcess.Face_alignment import FaceAligment, FaceType
from ProcessingFace.CoreProcess.Face_detector import FaceDetection
from ProcessingFace.CoreProcess.Face_landmarks import FaceLandmark
from ProcessingFace.CoreProcess.utlis import parse_face


class PipelineFace(object):
    def __init__(self, output_size):
        self.output_size = output_size

        # step 1
        self.detector = FaceDetection()

        # step 2: landmarks
        self.landmark = FaceLandmark()

        # step 3: alignment
        self.alignment = FaceAligment

    def __call__(self, img):
        """
            img: class ImageClass
        """
        list_face = []
        bboxes = self.detector.active(img)

        for bbox in bboxes:
            bbox = parse_face(bbox)

            landmarks = self.landmark.active(img, bbox)

            face_mat = self.alignment.active(landmarks[0], self.output_size, FaceType.FULL)

            face_img = cv2.warpAffine(img, face_mat, (self.output_size, self.output_size), cv2.INTER_CUBIC)

            list_face.append(face_img)

        return list_face
