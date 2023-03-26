import dtlpy as dl
import mediapipe as mp
import cv2


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe hands model',
                              init_inputs={'model_entity': dl.Model})
class MPHandsModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_hands = mp.solutions.hands

    def detect(self, model, image):
        # results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = model.process(image)
        if not results.multi_hand_landmarks:
            print('here')
            return []
        h, w, _ = image.shape

        def get_lm_px(landmarks, lm):
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

        ret = []
        for hand_landmarks in results.multi_hand_landmarks:
            print('here1')
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    ],
                    label='index'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    ],
                    label='middle'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_TIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_DIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_PIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_MCP),
                    ],
                    label='ring'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_TIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_DIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_PIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_MCP),
                    ],
                    label='pinky'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_TIP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_IP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_MCP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_CMC),
                    ],
                    label='thumb'
                ),
                'score': 0.5
            })
            ret.append({
                'annotation': dl.Polyline(
                    geo=[
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.WRIST),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.PINKY_MCP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.RING_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.INDEX_FINGER_MCP),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.WRIST),
                        get_lm_px(hand_landmarks.landmark, self.mp_hands.HandLandmark.THUMB_CMC),
                    ],
                    label='hand'
                ),
                'score': 0.5
            })
        print('here2')
        return ret

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as model:
            for image in batch:
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = self.detect(model, image)

                print("[INFO] uploading annotations...")
                image_annotations = dl.AnnotationCollection()

                # Draw face detections of each face.
                for detection in results:
                    image_annotations.add(annotation_definition=detection['annotation'],
                                          model_info={
                                              'name': 'MediaPipe',
                                              'confidence': detection['score']
                                          })
                batch_annotations.append(image_annotations)
        return batch_annotations
