# MediaPipe Models for Dataloop

This repository implements `ModelAdapter`s that uses pretrained models from Google's real time ML solution - MediaPipe.

You can read more about MediaPipe's capabilities and models [here](https://google.github.io/mediapipe/)

Each model has its own directory, with the (possible) following files:

| File             | Purpose                                    |
|------------------|--------------------------------------------|
| model_adapter.py | The `ModelAdapter` for the model           |
| service.py       | A `ServiceRunner` of the model for FaaS    |
| test.py          | A file to test out and play with the model |

The file create_model.py is responsible for uploading and deploying any of the `ModelAdapter`s  