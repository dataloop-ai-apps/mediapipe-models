import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class PoseModel(MediapipeModel):
    def detect_img(self, model, image):
        results = model.process(image)
        if not results.pose_landmarks:
            return []
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        def get_lm_px(lm_name):
            lm = getattr(mp.solutions.pose.PoseLandmark, lm_name)
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        return [
            {
                'annotation': dl.Point(*get_lm_px(kp), label=kp),
                'score': 0.5
            } for kp in self.get_labels()
        ]

    @property
    def model_cls(self):
        return mp.solutions.pose.Pose

    @staticmethod
    def get_name():
        return "pose"

    @staticmethod
    def get_default_model_configuration():
        return {
            'static_image_mode': True,
            'model_complexity': 2,
            'enable_segmentation': True,
            'min_detection_confidence': 0.5
        }

    @staticmethod
    def get_labels():
        return [
            'NOSE',
            'LEFT_EYE_INNER',
            'LEFT_EYE',
            'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER',
            'RIGHT_EYE',
            'RIGHT_EYE_OUTER',
            'LEFT_EAR',
            'RIGHT_EAR',
            'MOUTH_LEFT',
            'MOUTH_RIGHT',
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_ELBOW',
            'RIGHT_ELBOW',
            'LEFT_WRIST',
            'RIGHT_WRIST',
            'LEFT_PINKY',
            'RIGHT_PINKY',
            'LEFT_INDEX',
            'RIGHT_INDEX',
            'LEFT_THUMB',
            'RIGHT_THUMB',
            'LEFT_HIP',
            'RIGHT_HIP',
            'LEFT_KNEE',
            'RIGHT_KNEE',
            'LEFT_ANKLE',
            'RIGHT_ANKLE',
            'LEFT_HEEL',
            'RIGHT_HEEL',
            'LEFT_FOOT_INDEX',
            'RIGHT_FOOT_INDEX',
        ]

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.POINT
