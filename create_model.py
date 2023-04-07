import os
import dtlpy as dl
from typing import List, Type
from mediapipe_model import MediapipeModel
from model_adapter import MediapipeModelAdapter
from models import models


def upload_models(project_name, dataset_name, mp_models: List[Type[MediapipeModel]]):
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)

    codebase = project.codebases.pack(directory='./')
    metadata = dl.Package.get_ml_metadata(cls=MediapipeModelAdapter, default_configuration={})
    module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name=f'mediapipe-package',
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

    for mp_model in mp_models:
        model = package.models.create(model_name=f'mp-{mp_model.get_name()}-model',
                                      description=f'mediapipe {mp_model.get_name()} model',
                                      tags=['pretrained'],
                                      dataset_id=dataset.id,
                                      project_id=package.project.id,
                                      configuration={
                                          "model_name": mp_model.get_name(),
                                          "model_config": mp_model.get_default_model_configuration()
                                      },
                                      model_artifacts=[],
                                      labels=mp_model.get_labels(),
                                      output_type=mp_model.get_output_type()
                                      )

        model.status = 'trained'
        model.update()
        model.deploy()


def main():
    if dl.token_expired():
        dl.login()
    upload_models('My First Project', 'faces', mp_models=models)


if __name__ == '__main__':
    main()
