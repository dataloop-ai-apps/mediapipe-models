from adapters.face.face_model import FaceModel
from adapters.face_mesh.face_mesh_model import FaceMeshModel
from adapters.hands.hands_model import HandsModel
from adapters.holistic.holistic_model import HolisticModel
from adapters.objectron.objectron_model import ObjectronModel
from adapters.pose.pose_model import PoseModel
from adapters.selfie_seg.selfie_seg_model import SelfieSegmentationModel

models = [
    FaceModel,
    FaceMeshModel,
    HandsModel,
    HolisticModel,
    ObjectronModel,
    PoseModel,
    SelfieSegmentationModel
]
