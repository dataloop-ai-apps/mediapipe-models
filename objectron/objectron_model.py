import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class ObjectronModel(MediapipeModel):
    def detect_img(self, model, image):
        results = model.process(image)
        if not results.detected_objects:
            return []
        h, w, _ = image.shape
        ret = []
        for detection in results.detected_objects:
            norm_px = [(round(lm.x * w), round(lm.y * h)) for lm in detection.landmarks_2d.landmark]
            ret.append({
                'annotation': dl.Cube(
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
                'score': 0.5
            })
        return ret

    @property
    def model_cls(self):
        return mp.solutions.objectron.Objectron

    @staticmethod
    def get_name():
        return "objectron"

    @staticmethod
    def get_default_model_configuration():
        return {
            'static_image_mode': True,
            'max_num_objects': 5,
            'min_detection_confidence': 0.5,
            'model_name': 'Shoe'
        }

    @staticmethod
    def get_labels():
        return ['shoe']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.CUBE
