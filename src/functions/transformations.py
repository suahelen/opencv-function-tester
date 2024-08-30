import cv2
from .enums import Interpolation
from .function import CvFunction


class Resize(CvFunction):
    @staticmethod
    def process(image, fx, fy, interpolation=Interpolation.INTER_LINEAR):
        return cv2.resize(image, None, fx=fx, fy=fy, interpolation=interpolation.value)

    @staticmethod
    def get_params():
        return {
            "fx": (0.1, 10.0, 1.0, 0.1),  # Scale factor along the horizontal axis
            "fy": (0.1, 10.0, 1.0, 0.1),  # Scale factor along the vertical axis
            "interpolation": list(Interpolation),
        }
