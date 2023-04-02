import dtlpy as dl
import mediapipe as mp
from face.face_model import FaceModel
from face_mesh.face_mesh_model import FaceMeshModel
from hands.hands_model import HandsModel
from holistic.holistic_model import HolisticModel
from objectron.objectron_model import ObjectronModel
from pose.pose_model import PoseModel
from selfie_seg.selfie_seg_model import SelfieSegmentationModel


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for MediaPipe models',
                              init_inputs={'model_entity': dl.Model})
class MediapipeModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        models = [
            FaceModel,
            FaceMeshModel,
            HandsModel,
            HolisticModel,
            ObjectronModel,
            PoseModel,
            SelfieSegmentationModel
        ]
        self.mp_model = None
        for model in models:
            if self.configuration["model_name"] == model.get_name():
                self.mp_model = model(**self.configuration["model_config"])
                break
        else:
            print("WARNING: invalid model_name in configuration")

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        for results in self.mp_model.detect(batch):

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
