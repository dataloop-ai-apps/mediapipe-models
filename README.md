# MediaPipe Models for Dataloop

This repository implements `ModelAdapter`s that uses pretrained models from Google's real time ML solution - MediaPipe.

You can read more about MediaPipe's capabilities and models [here](https://google.github.io/mediapipe/)

## How to create a model
### 1. Create a `MediapipeModel` class

Each model has its own directory, with the following files:

| File             | Purpose                                    |
|------------------|--------------------------------------------|
| <name\>_model.py | The `MediapipeModel` wrapper for the model |
| test.py          | A file to test out and play with the model |

So the first step is to create the directory and file for the model.
In the file, create a class that inherits from `MediapipeModel` (defined in `mediapipe_model.py`).

The class should override all the abstract methods of `MediapipeModel`:

| Decorators | Method           | Input                                    | Returns |
|---|------------------|--------------------------------------------|---|
|  | detect_img       | `model`: the model of Mediapipe's SDK. <br> `image`: image to preform inference on. | A list of dictionaries of the form {'annotation': dl.Annotation, 'score': float} of all annotations |
| @property | model_cls          | - | the model class of Mediapipe's SDK. For example, `mp.solutions.face_detection.FaceDetection`|
| @staticmethod | get_name          | - | A string that represent the model. Will appear in Dataloop in the full name of the model as "mp-<name\>-model" |
| @staticmethod | get_default_model_configuration | - | A dictionary of the inputs for `model_cls`|
| @staticmethod | get_labels | - | A list of strings for all the possible labels of the model |
| @staticmethod | get_output_type | - | A `dl.AnnotationType` of the model. Must match the type that `detect_img` returns in `annotation`. |

### 2. Add your model to `models`

In the file `models.py`, import your class and add it to the `models` list.

You can add and remove models from the list as you wish - these are the models the will be uploaded to your project.

### 3. Upload the models

In the file `create_model.py`, inside the `main` function, pass your project name and dataset name (with the proper recipe).

 
This will upload and deploy the `ModelAdapter` (defined in `model_adapter.py`) that can call any of the `MediapipeModel`s

Run `create_model.py`, login to Dataloop if necessary, and you're good to go! check out your models under "Model Management"
and feel free to test them out under the "Test" tab.
