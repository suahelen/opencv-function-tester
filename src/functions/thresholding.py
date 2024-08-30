import cv2
from .enums import ThresholdType
from .function import CvFunction


class Threshold(CvFunction):
    @staticmethod
    def process(image, thresh, maxval, thresh_type):
        _, processed = cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), thresh, maxval, thresh_type.value
        )
        return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def get_params():
        return {
            "thresh": (0, 255, 127, 1),
            "maxval": (0, 255, 255, 1),
            "thresh_type": list(ThresholdType),
        }
