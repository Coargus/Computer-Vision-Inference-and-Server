import abc


class VisionLanguageModelBase(abc.ABC):
    """Vision Language Model Base class."""

    model: any

    def __init__(self, device: str) -> None:
        super().__init__(device)

    @abc.abstractmethod
    def infer_with_image(self, vision_input: any, language_input: str) -> any:
        raise NotImplementedError

    @abc.abstractmethod
    def infer_with_video(self, vision_input: any, language_input: str) -> any:
        raise NotImplementedError
