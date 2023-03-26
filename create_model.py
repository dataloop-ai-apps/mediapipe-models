import dtlpy as dl
import os


def upload_model(model_name, model_cls, model_out, model_lab):
    os.chdir(f'./{model_name.replace("-", "_")}')
    project = dl.projects.get(project_name='My First Project')
    dataset = project.datasets.get(dataset_name='faces')

    codebase = project.codebases.pack(directory='./')
    metadata = dl.Package.get_ml_metadata(cls=model_cls,
                                          default_configuration={},
                                          output_type=model_out)
    module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name=f'mediapipe-package-{model_name}',
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    codebase=codebase,
                                    modules=[module],
                                    is_global=False,
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        runner_image='docker.io/yakirinven/mediapipe_image',
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)

    model = package.models.create(model_name=f'mp-{model_name}-model',
                                  description=f'mediapipe {model_name} model',
                                  tags=['pretrained'],
                                  dataset_id=dataset.id,
                                  project_id=package.project.id,
                                  configuration={},
                                  model_artifacts=[],
                                  labels=model_lab
                                  )

    model.status = 'trained'
    model.update()
    model.deploy()
    os.chdir(f'..')


def upload_models(model_names, model_clss, model_outs, model_labs):
    # os.chdir(f'./{model_name.replace("-", "_")}')
    project = dl.projects.get(project_name='My First Project')
    dataset = project.datasets.get(dataset_name='faces')

    codebase = project.codebases.pack(directory='./')
    metadata = []

    for model_cls, model_out in zip(model_clss, model_outs):
        metadata.append(dl.Package.get_ml_metadata(cls=model_cls,
                                                   default_configuration={},
                                                   output_type=model_out))
    module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name=f'mediapipe-package-{model_name}',
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    codebase=codebase,
                                    modules=[module],
                                    is_global=False,
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        runner_image='docker.io/yakirinven/mediapipe_image',
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)

    model = package.models.create(model_name=f'mp-{model_name}-model',
                                  description=f'mediapipe {model_name} model',
                                  tags=['pretrained'],
                                  dataset_id=dataset.id,
                                  project_id=package.project.id,
                                  configuration={},
                                  model_artifacts=[],
                                  labels=model_lab
                                  )

    model.status = 'trained'
    model.update()
    model.deploy()
    os.chdir(f'..')


def main():
    pass
    key_points = [
        'NOSE',
        'LEFT_EYE_INNER',
        'LEFT_EYE',
        'LEFT_EYE_OUTER',
        'RIGHT_EYE_INNER',
        'RIGHT_EYE',
        'RIGHT_EYE_OUTER',
        'LEFT_EAR',
        'RIGHT_EAR',
        'MOUTH_LEFT',
        'MOUTH_RIGHT',
        'LEFT_SHOULDER',
        'RIGHT_SHOULDER',
        'LEFT_ELBOW',
        'RIGHT_ELBOW',
        'LEFT_WRIST',
        'RIGHT_WRIST',
        'LEFT_PINKY',
        'RIGHT_PINKY',
        'LEFT_INDEX',
        'RIGHT_INDEX',
        'LEFT_THUMB',
        'RIGHT_THUMB',
        'LEFT_HIP',
        'RIGHT_HIP',
        'LEFT_KNEE',
        'RIGHT_KNEE',
        'LEFT_ANKLE',
        'RIGHT_ANKLE',
        'LEFT_HEEL',
        'RIGHT_HEEL',
        'LEFT_FOOT_INDEX',
        'RIGHT_FOOT_INDEX',
    ]
    # #####
    # from face.model_adapter import MPFaceModelAdapter
    # upload_model('face', MPFaceModelAdapter, dl.AnnotationType.BOX, ['person'])
    # # #####
    # from objectron.model_adapter import MPObjectronShoeModelAdapter
    # upload_model('objectron', MPObjectronShoeModelAdapter, dl.AnnotationType.CUBE, ['shoe'])
    # # #####
    # from hands.model_adapter import MPHandsModelAdapter
    # upload_model('hands', MPHandsModelAdapter, dl.AnnotationType.POLYLINE,
    #              ['hand', 'thumb', 'pinky', 'ring', 'middle', 'index'])
    # # #####
    # from pose.model_adapter import MPPoseModelAdapter
    # upload_model('pose', MPPoseModelAdapter, dl.AnnotationType.POSE, key_points)
    # # #####
    from face_mesh.model_adapter import MPFaceMeshModelAdapter
    upload_model('face-mesh', MPFaceMeshModelAdapter, dl.AnnotationType.POLYLINE,
                 ['tessellation', 'irises', 'contours'])
    # # #####
    # from holistic.model_adapter import MPHolisticModelAdapter
    # upload_model('holistic', MPHolisticModelAdapter, dl.AnnotationType.POLYLINE,
    #              ['tessellation', 'contours', 'pose', 'hand'])
    # # #####
    # from selfie_seg.model_adapter import MPSelfieSegmentationModelAdapter
    # upload_model('selfie-seg', MPSelfieSegmentationModelAdapter, dl.AnnotationType.SEGMENTATION, ['selfie'])


if __name__ == '__main__':
    main()
