import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class SelfieSegmentationModel(MediapipeModel):
    def detect_img(self, model, image):
        results = model.process(image)
        return [{
            'annotation': dl.Segmentation(geo=results.segmentation_mask, label='selfie'),
            'score': 0.5
        }]

    @property
    def model_cls(self):
        return mp.solutions.selfie_segmentation.SelfieSegmentation

    @staticmethod
    def get_name():
        return "selfie-seg"

    @staticmethod
    def get_default_model_configuration():
        return {
            'model_selection': 0
        }

    @staticmethod
    def get_labels():
        return ['selfie']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.SEGMENTATION
