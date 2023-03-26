import dtlpy as dl
import mediapipe as mp
import cv2


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe face mesh model',
                              init_inputs={'model_entity': dl.Model})
class MPFaceMeshModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_face_mesh = mp.solutions.face_mesh

    def detect(self, model, image):
        results = model.process(image)
        if not results.multi_face_landmarks:
            return []
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        for face_landmarks in results.multi_face_landmarks:
            for con in self.mp_face_mesh.FACEMESH_TESSELATION:
                if con in self.mp_face_mesh.FACEMESH_CONTOURS or con in self.mp_face_mesh.FACEMESH_IRISES:
                    continue
                ret.append(dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                        get_lm_px(face_landmarks.landmark, con[1])], 'tessellation'))
            for con in self.mp_face_mesh.FACEMESH_CONTOURS:
                if con in self.mp_face_mesh.FACEMESH_IRISES:
                    continue
                ret.append(dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                        get_lm_px(face_landmarks.landmark, con[1])], 'contours'))
            for con in self.mp_face_mesh.FACEMESH_IRISES:
                ret.append(dl.Polyline([get_lm_px(face_landmarks.landmark, con[0]),
                                        get_lm_px(face_landmarks.landmark, con[1])], 'irises'))
        return ret

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as model:
            for image in batch:
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = self.detect(model, image)

                print("[INFO] uploading annotations...")
                image_annotations = dl.AnnotationCollection()

                # Draw face detections of each face.
                for annotation in results:
                    image_annotations.add(annotation_definition=annotation,
                                          model_info={
                                              'name': 'MediaPipe',
                                              'confidence': 0.5
                                          })
                batch_annotations.append(image_annotations)
        return batch_annotations
