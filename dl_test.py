import dtlpy as dl

# login
if dl.token_expired():
    ans = dl.login()

project = dl.projects.get(project_name="My First Project")
dataset = project.datasets.get(dataset_name="faces")

recipe = dataset.recipes.list()[0]
pose_template_id = recipe.get_annotation_template_id('pose')

image_annotations = dl.AnnotationCollection()
image_annotations.add(annotation_definition=dl.Pose(label='my_parent_label',
                                                    template_id=pose_template_id),

                      model_info={
                          'name': 'MediaPipe',
                          'confidence': 0.5
                      })
image_annotations[0].id = 5

image_annotations.add(annotation_definition=dl.Point(0,0,label='NOSE'),
                      parent_id=5,
                      model_info={
                          'name': 'MediaPipe',
                          'confidence': 0.5
                      })

print(image_annotations.to_json())
