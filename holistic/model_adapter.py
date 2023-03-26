import dtlpy as dl
import mediapipe as mp


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe holistic model',
                              init_inputs={'model_entity': dl.Model})
class MPHolisticModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_holistic = mp.solutions.holistic

    def detect(self, model, image):
        results = model.process(image)
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        if results.face_landmarks:
            for con in self.mp_holistic.FACEMESH_TESSELATION:
                ret.append(dl.Polyline([get_lm_px(results.face_landmarks.landmark, con[0]),
                                        get_lm_px(results.face_landmarks.landmark, con[1])], 'tessellation'))
            for con in self.mp_holistic.FACEMESH_CONTOURS:
                ret.append(dl.Polyline([get_lm_px(results.face_landmarks.landmark, con[0]),
                                        get_lm_px(results.face_landmarks.landmark, con[1])], 'contours'))
        if results.pose_landmarks:
            for con in self.mp_holistic.POSE_CONNECTIONS:
                ret.append(dl.Polyline([get_lm_px(results.pose_landmarks.landmark, con[0]),
                                        get_lm_px(results.pose_landmarks.landmark, con[1])], 'pose'))
        if results.left_hand_landmarks:
            for con in self.mp_holistic.HAND_CONNECTIONS:
                ret.append(dl.Polyline([get_lm_px(results.left_hand_landmarks.landmark, con[0]),
                                        get_lm_px(results.left_hand_landmarks.landmark, con[1])], 'hand'))
        if results.right_hand_landmarks:
            for con in self.mp_holistic.HAND_CONNECTIONS:
                ret.append(dl.Polyline([get_lm_px(results.right_hand_landmarks.landmark, con[0]),
                                        get_lm_px(results.right_hand_landmarks.landmark, con[1])], 'hand'))
        return ret

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                refine_face_landmarks=True) as model:
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
