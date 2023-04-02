import dtlpy as dl
import mediapipe as mp
from mediapipe_model import MediapipeModel


class FaceMeshModel(MediapipeModel):
    def detect_img(self, model, image):
        mp_face_mesh = mp.solutions.face_mesh
        results = model.process(image)
        if not results.multi_face_landmarks:
            return []
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        for face_landmarks in results.multi_face_landmarks:
            for con in mp_face_mesh.FACEMESH_TESSELATION:
                if con in mp_face_mesh.FACEMESH_CONTOURS or con in mp_face_mesh.FACEMESH_IRISES:
                    continue
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                               get_lm_px(face_landmarks.landmark, con[1])], 'tessellation'),
                    'score': 0.5
                })
            for con in mp_face_mesh.FACEMESH_CONTOURS:
                if con in mp_face_mesh.FACEMESH_IRISES:
                    continue
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                               get_lm_px(face_landmarks.landmark, con[1])], 'contours'),
                    'score': 0.5
                })
            for con in mp_face_mesh.FACEMESH_IRISES:
                ret.append({
                    'annotation': dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                               get_lm_px(face_landmarks.landmark, con[1])], 'irises'),
                    'score': 0.5
                })
        return ret

    @property
    def model_cls(self):
        return mp.solutions.face_mesh.FaceMesh

    @staticmethod
    def get_name():
        return "face-mesh"

    @staticmethod
    def get_default_model_configuration():
        return {
            'static_image_mode': True,
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': 0.5
        }

    @staticmethod
    def get_labels():
        return ['tessellation', 'irises', 'contours']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.POLYLINE
