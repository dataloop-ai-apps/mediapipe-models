import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class HolisticModel(MediapipeModel):
    def detect_img(self, model, image):
        mp_holistic = mp.solutions.holistic
        results = model.process(image)
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        if results.face_landmarks:
            for con in mp_holistic.FACEMESH_TESSELATION:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(results.face_landmarks.landmark, con[0]),
                                               get_lm_px(results.face_landmarks.landmark, con[1])], 'tessellation'),
                    'score': 0.5
                })
            for con in mp_holistic.FACEMESH_CONTOURS:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(results.face_landmarks.landmark, con[0]),
                                               get_lm_px(results.face_landmarks.landmark, con[1])], 'contours'),
                    'score': 0.5
                })
        if results.pose_landmarks:
            for con in mp_holistic.POSE_CONNECTIONS:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(results.pose_landmarks.landmark, con[0]),
                                               get_lm_px(results.pose_landmarks.landmark, con[1])], 'pose'),
                    'score': 0.5
                })
        if results.left_hand_landmarks:
            for con in mp_holistic.HAND_CONNECTIONS:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(results.left_hand_landmarks.landmark, con[0]),
                                               get_lm_px(results.left_hand_landmarks.landmark, con[1])], 'hand'),
                    'score': 0.5
                })
        if results.right_hand_landmarks:
            for con in mp_holistic.HAND_CONNECTIONS:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(results.right_hand_landmarks.landmark, con[0]),
                                               get_lm_px(results.right_hand_landmarks.landmark, con[1])], 'hand'),
                    'score': 0.5
                })
        return ret

    @property
    def model_cls(self):
        return mp.solutions.holistic.Holistic

    @staticmethod
    def get_name():
        return "holistic"

    @staticmethod
    def get_default_model_configuration():
        return {
            'static_image_mode': True,
            'model_complexity': 2,
            'enable_segmentation': True,
            'refine_face_landmarks': True
        }

    @staticmethod
    def get_labels():
        return ['tessellation', 'contours', 'pose', 'hand']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.POLYLINE
