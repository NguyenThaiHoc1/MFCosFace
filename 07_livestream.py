import cv2
from pathlib import Path
from ProcessingFace.Pipeline.pipelineface import PipelineFace

if __name__ == '__main__':

    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)

    pipeline = PipelineFace(output_size=128)

    person_name = 'tam-tl'

    path_register = Path('/Volumes/Ventoy/Data/face_register') / person_name
    count = 0

    while cap.isOpened():
        success, img = cap.read()
        if success:

            face = pipeline(img.copy())

            if len(face) == 0:
                continue
            path_filename = path_register / f'{person_name}_{count}.jpg'
            cv2.imshow("Result", face[0])
            cv2.imwrite(str(path_filename), face[0])
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
