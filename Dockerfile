FROM docker.io/dataloopai/dtlpy-agent:cpu.py3.10.opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
RUN pip3 install mediapipe>=0.10.21

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/mediapipe-models:0.0.1 -f ./Dockerfile  .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/mediapipe-models:0.0.1 bash
# docker push gcr.io/viewo-g/piper/agent/runner/apps/mediapipe-models:0.0.1