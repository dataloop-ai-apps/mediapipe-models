import dtlpy as dl
import mediapipe as mp


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe selfie segmentation model',
                              init_inputs={'model_entity': dl.Model})
class MPSelfieSegmentationModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation

    def detect(self, model, image):
        results = model.process(image)
        ret = [dl.Segmentation(geo=results.segmentation_mask, label='selfie')]
        return ret

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as model:
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
