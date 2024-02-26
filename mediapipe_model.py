from abc import ABC, abstractmethod


class MediapipeModel(ABC):
    def __init__(self, **model_config):
        self.model_config = model_config

    @abstractmethod
    def detect_img(self, model, image):
        pass

    def detect(self, batch):
        with self.model_cls(**self.model_config) as model:
            return [self.detect_img(model, image) for image in batch]

    @property
    @abstractmethod
    def model_cls(self):
        pass

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @staticmethod
    @abstractmethod
    def get_default_model_configuration():
        pass

    @staticmethod
    @abstractmethod
    def get_labels():
        pass

    @staticmethod
    @abstractmethod
    def get_output_type():
        pass
