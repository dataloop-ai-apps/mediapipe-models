import dtlpy as dl
import mediapipe as mp


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for mediapipe pose model',
                              init_inputs={'model_entity': dl.Model})
class MPPoseModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.mp_pose = mp.solutions.pose

        project = dl.projects.get(project_name="My First Project")
        dataset = project.datasets.get(dataset_name="faces")
        recipe = dataset.recipes.list()[0]
        self.pose_template_id = recipe.get_annotation_template_id('pose')

    def detect(self, model, image):
        # results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = model.process(image)
        if not results.pose_landmarks:
            return []
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        def get_lm_px(lm_name):
            lm = getattr(self.mp_pose.PoseLandmark, lm_name)
            return [round(landmarks[lm].x * w), round(landmarks[lm].y * h)]

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
        ret = [
            {
                'annotation': dl.Point(*get_lm_px(kp), label=kp),
                'score': 0.5
            } for kp in key_points
        ]
        return ret

    def predict(self, batch, **kwargs):
        print('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []

        with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as model:
            for image in batch:
                results = self.detect(model, image)

                print("[INFO] uploading annotations...")
                image_annotations = dl.AnnotationCollection()
                image_annotations.add(annotation_definition=dl.Pose(label='pose',
                                                                    template_id=self.pose_template_id),

                                      model_info={
                                          'name': 'MediaPipe',
                                          'confidence': 0.5
                                      })
                image_annotations[0].id = 5

                for detection in results:
                    image_annotations.add(annotation_definition=detection['annotation'],
                                          parent_id=5,
                                          model_info={
                                              'name': 'MediaPipe',
                                              'confidence': detection['score']
                                          })
                batch_annotations.append(image_annotations)
        return batch_annotations
