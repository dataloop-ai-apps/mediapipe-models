import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class HandsModel(MediapipeModel):
    def detect_img(self, model, image):
        mp_hands = mp.solutions.hands
        results = model.process(image)
        if not results.multi_hand_landmarks:
            return []
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        for hand_landmarks in results.multi_hand_landmarks:
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    ],
                    label='index'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    ],
                    label='middle'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_MCP),
                    ],
                    label='ring'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_TIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_DIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_PIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_MCP),
                    ],
                    label='pinky'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_TIP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_IP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_MCP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_CMC),
                    ],
                    label='thumb'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.WRIST),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.PINKY_MCP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.WRIST),
                        get_lm_px(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_CMC),
                    ],
                    label='hand'
                ),
                'score': 0.5
            })
        return ret

    @property
    def model_cls(self):
        return mp.solutions.hands.Hands

    @staticmethod
    def get_name():
        return "hands"

    @staticmethod
    def get_default_model_configuration():
        return {
            'static_image_mode': True,
            'max_num_hands': 2,
            'min_detection_confidence': 0.5
        }

    @staticmethod
    def get_labels():
        return ['hand', 'thumb', 'pinky', 'ring', 'middle', 'index']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.POLYLINE
