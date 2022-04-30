from mtcnn import MTCNN


class FaceDetection(object):
    detector = None

    def __init__(self, set_detector='mtcnn'):
        if set_detector == 'mtcnn':
            self.detector = MTCNN()
        else:
            print(f"We will update soon !")

    def active(self, image):
        """
            run detect faces in image
            -------------------------
            image: numpy_array
        """
        return [bbox['box'] for bbox in self.detector.detect_faces(image)]
