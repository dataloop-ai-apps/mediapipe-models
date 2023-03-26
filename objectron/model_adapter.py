import dtlpy as dl
import mediapipe as mp
import cv2


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe objectron shoe model',
                              init_inputs={'model_entity': dl.Model})
class MPObjectronShoeModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_objectron = mp.solutions.objectron

    def detect(self, model, image):
        results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_objectron.Objectron(static_image_mode=True,
                                         max_num_objects=5,
                                         min_detection_confidence=0.5,
                                         model_name='Shoe') as model:
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
