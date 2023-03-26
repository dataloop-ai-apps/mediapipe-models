import cv2
import os
import mediapipe as mp
import dtlpy as dl


class ServiceRunner:
    def __init__(self):
        self.mp_model = mp.solutions.hands
        # self.mp_drawing = mp.solutions.drawing_utils

    def build_hand_ann(self, img, builder, landmarks):
        h, w, _ = img.shape

        def get_lm_px(lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.INDEX_FINGER_TIP),
                    get_lm_px(self.mp_model.HandLandmark.INDEX_FINGER_DIP),
                    get_lm_px(self.mp_model.HandLandmark.INDEX_FINGER_PIP),
                    get_lm_px(self.mp_model.HandLandmark.INDEX_FINGER_MCP),
                ],
                label='index'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.MIDDLE_FINGER_TIP),
                    get_lm_px(self.mp_model.HandLandmark.MIDDLE_FINGER_DIP),
                    get_lm_px(self.mp_model.HandLandmark.MIDDLE_FINGER_PIP),
                    get_lm_px(self.mp_model.HandLandmark.MIDDLE_FINGER_MCP),
                ],
                label='middle'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.RING_FINGER_TIP),
                    get_lm_px(self.mp_model.HandLandmark.RING_FINGER_DIP),
                    get_lm_px(self.mp_model.HandLandmark.RING_FINGER_PIP),
                    get_lm_px(self.mp_model.HandLandmark.RING_FINGER_MCP),
                ],
                label='ring'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.PINKY_TIP),
                    get_lm_px(self.mp_model.HandLandmark.PINKY_DIP),
                    get_lm_px(self.mp_model.HandLandmark.PINKY_PIP),
                    get_lm_px(self.mp_model.HandLandmark.PINKY_MCP),
                ],
                label='pinky'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.THUMB_TIP),
                    get_lm_px(self.mp_model.HandLandmark.THUMB_IP),
                    get_lm_px(self.mp_model.HandLandmark.THUMB_MCP),
                    get_lm_px(self.mp_model.HandLandmark.THUMB_CMC),
                ],
                label='thumb'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

        builder.add(
            annotation_definition=dl.Polyline(
                geo=[
                    get_lm_px(self.mp_model.HandLandmark.WRIST),
                    get_lm_px(self.mp_model.HandLandmark.PINKY_MCP),
                    get_lm_px(self.mp_model.HandLandmark.RING_FINGER_MCP),
                    get_lm_px(self.mp_model.HandLandmark.MIDDLE_FINGER_MCP),
                    get_lm_px(self.mp_model.HandLandmark.INDEX_FINGER_MCP),
                    get_lm_px(self.mp_model.HandLandmark.WRIST),
                    get_lm_px(self.mp_model.HandLandmark.THUMB_CMC),
                ],
                label='hand'
            ),
            model_info={
                'name': 'MediaPipe',
                'confidence': 0.5
            }
        )

    def detect(self, item):
        print("[INFO] downloading image...")
        filename = item.download()
        try:
            with self.mp_model.Hands(
                    static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
                image = cv2.imread(filename)
                # for correct handedness: (remember to also flip results)
                # image = cv2.flip(cv2.imread(filename), 1)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                print("[INFO] uploading annotations...")
                builder = item.annotations.builder()

                if not results.multi_hand_landmarks:
                    return

                for hand_landmarks in results.multi_hand_landmarks:
                    self.build_hand_ann(image, builder, hand_landmarks.landmark)

                # upload annotations
                builder.upload()
                print("[INFO] Done!")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            os.remove(filename)
