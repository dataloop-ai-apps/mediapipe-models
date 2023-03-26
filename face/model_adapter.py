import dtlpy as dl
import mediapipe as mp


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe face detection model',
                              init_inputs={'model_entity': dl.Model})
class MPFaceModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_face_detection = mp.solutions.face_detection

    def get_absolute_pixels(self, img, rbb):
        h, w, _ = img.shape
        xs = round(rbb.xmin * w)
        ys = round(rbb.ymin * h)
        return xs, ys, xs + round(rbb.width * w), ys + round(rbb.height * h)

    def detect(self, model, image):
        results = model.process(image)
        ret = []
        for detection in results.detections or []:
            (startX, startY, endX, endY) = \
                self.get_absolute_pixels(image, detection.location_data.relative_bounding_box)
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

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_face_detection.FaceDetection(model_selection=1,
                                                  min_detection_confidence=0.5) as model:
            for image in batch:
                results = self.detect(model, image)

                print("[INFO] uploading annotations...")
                image_annotations = dl.AnnotationCollection()

                for detection in results:
                    image_annotations.add(annotation_definition=detection['annotation'],
                                          model_info={
                                              'name': 'MediaPipe',
                                              'confidence': detection['score']
                                          })
                batch_annotations.append(image_annotations)
        return batch_annotations
