import cv2
import os
import mediapipe as mp
import dtlpy as dl


class ServiceRunner:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        # self.mp_drawing = mp.solutions.drawing_utils

    def get_absolute_pixels(self, img, rbb):
        h, w, _ = img.shape
        xs = round(rbb.xmin * w)
        ys = round(rbb.ymin * h)
        return xs, ys, xs + round(rbb.width * w), ys + round(rbb.height * h)

    def detect(self, item):
        print("[INFO] downloading image...")
        filename = item.download()
        try:
            with self.mp_face_detection.FaceDetection(model_selection=1,
                                                      min_detection_confidence=0.5) as face_detection:
                image = cv2.imread(filename)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                print("[INFO] uploading annotations...")
                builder = item.annotations.builder()

                # Draw face detections of each face.
                if not results.detections:
                    return
                for detection in results.detections:
                    (startX, startY, endX, endY) = \
                        self.get_absolute_pixels(image, detection.location_data.relative_bounding_box)
                    # draw the bounding box of the face along with the associated
                    # probability
                    builder.add(
                        annotation_definition=dl.Box(
                            top=startY,
                            left=startX,
                            right=endX,
                            bottom=endY,
                            label='person'
                        ),
                        model_info={
                            'name': 'MediaPipe',
                            'confidence': detection.score[0]
                        }
                    )
                    # upload annotations
                builder.upload()
                print("[INFO] Done!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            os.remove(filename)
