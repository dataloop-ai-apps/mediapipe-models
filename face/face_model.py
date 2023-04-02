import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


def get_absolute_pixels(img, rbb):
    h, w, _ = img.shape
    xs = round(rbb.xmin * w)
    ys = round(rbb.ymin * h)
    return xs, ys, xs + round(rbb.width * w), ys + round(rbb.height * h)


class FaceModel(MediapipeModel):
    def detect_img(self, model, image):
        results = model.process(image)
        ret = []
        for detection in results.detections or []:
            (startX, startY, endX, endY) = \
                get_absolute_pixels(image, detection.location_data.relative_bounding_box)
            ret.append({
                'annotation': dl.Box(
                    top=startY,
                    left=startX,
                    right=endX,
                    bottom=endY,
                    label='person'
                ),
                'score': detection.score[0]
            })
        return ret

    @property
    def model_cls(self):
        return mp.solutions.face_detection.FaceDetection

    @staticmethod
    def get_name():
        return "face"

    @staticmethod
    def get_default_model_configuration():
        return {
            'model_selection': 1,
            'min_detection_confidence': 0.5
        }

    @staticmethod
    def get_labels():
        return ['person']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.BOX
