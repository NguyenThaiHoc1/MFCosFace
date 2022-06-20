from ProcessingFace.CoreProcess.Face_detector import FaceDetection
from ProcessingFace.CoreProcess.Face_landmarks import FaceLandmark, shape_to_landmarks
from ProcessingFace.CoreProcess.utlis import get_six_points, mask_face, parse_face, rect_to_bb


class PipelineMaskTheFace(object):
    def __init__(self):

        # step 1: detector
        self.detector = FaceDetection()

        # step 2: landmark
        self.landmark = FaceLandmark(landmark='dlib')

        # step 3: convert landmark ==> dictionary details
        self.convert_landmark = shape_to_landmarks

        # step 4: Get six points
        self.get_six_points = get_six_points

        # step 5: Assign mask to face
        self.maskface = mask_face

    def active(self, img, mask_type="surgical", dict_para={'color': "#0473e2", 'color_weight': 0.5,
                                                           'pattern': "",
                                                           'pattern_weight': 0.5}):

        # step 1:
        origin_image = img.copy()
        bboxes = self.detector.active(img)

        masked_images = []
        mask_binary_array = []
        mask = []
        for bbox in bboxes:

            bbox = parse_face(bbox)  # top, left, bottom, right

            landmarks = self.landmark.active(img, bbox)

            new_shape_landmark = shape_to_landmarks(landmarks[0])

            face_location = rect_to_bb(bbox)

            # get six point
            six_points_on_face, angle = self.get_six_points(new_shape_landmark, img)

            if mask_type != "all":
                image, mask_binary = self.maskface(img, face_location,
                                                   six_points_on_face, angle,
                                                   dict_para, mask_type=mask_type)

                # compress to face tight
                masked_images.append(image)
                mask_binary_array.append(mask_binary)
                mask.append(mask_type)

        return masked_images, mask, mask_binary_array, origin_image
