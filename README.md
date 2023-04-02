# MediaPipe Models for Dataloop

This repository implements `ModelAdapter`s that uses pretrained models from Google's real time ML solution - MediaPipe.

You can read more about MediaPipe's capabilities and models [here](https://google.github.io/mediapipe/)

Each model has its own directory, with the (possible) following files:

| File             | Purpose                                    |
|------------------|--------------------------------------------|
| <name\>_model.py | The `MediapipeModel` wrapper for the model |
| test.py          | A file to test out and play with the model |

The file model_adapter.py is contains the `ModelAdapter` that can call any of the `MediapipeModel`s according to configuration

The file create_model.py is responsible for uploading and deploying the `ModelAdapter`  
