import cv2
import os
import mediapipe as mp
import dtlpy as dl

class ServiceRunner:
    def __init__(self):
        self.mp_objectron = mp.solutions.objectron
        # self.mp_drawing = mp.solutions.drawing_utils

        
    def detect(self, item: dl.Item):
        print("[INFO] downloading image...")
        filename = item.download()
        try:
            with self.mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
                image = cv2.imread(filename)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Draw face detections of each face.
                if not results.detected_objects:
                    return
                
                print("[INFO] uploading annotations...")
                builder = item.annotations.builder()
                
                h, w, _ = image.shape
                for detection in results.detected_objects:
                    norm_px = [(round(lm.x * w), round(lm.y * h)) for lm in detection.landmarks_2d.landmark]
                    builder.add(
                        annotation_definition=dl.Cube(
                            label='shoe',
                            front_bl=norm_px[2],
                            front_tl=norm_px[4],
                            front_br=norm_px[6],
                            front_tr=norm_px[8],
                            back_bl=norm_px[1],
                            back_tl=norm_px[3],
                            back_br=norm_px[5],
                            back_tr=norm_px[7],
                        ),
                        model_info={
                            'name': 'MediaPipe',
                            'confidence': 0.5
                        }
                    )
                builder.upload()
                print("[INFO] Done!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            os.remove(filename)
            pass
