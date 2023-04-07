from face.face_model import FaceModel
from face_mesh.face_mesh_model import FaceMeshModel
from hands.hands_model import HandsModel
from holistic.holistic_model import HolisticModel
from objectron.objectron_model import ObjectronModel
from pose.pose_model import PoseModel
from selfie_seg.selfie_seg_model import SelfieSegmentationModel

models = [
    FaceModel,
    FaceMeshModel,
    HandsModel,
    HolisticModel,
    ObjectronModel,
    PoseModel,
    SelfieSegmentationModel
]
