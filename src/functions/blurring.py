import cv2
from .enums import BorderType
from .function import CvFunction


class GaussianBlur(CvFunction):
    @staticmethod
    def process(image, ksize, sigmaX, sigmaY=0, borderType=BorderType.BORDER_DEFAULT):
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX, sigmaY, borderType.value)

    @staticmethod
    def get_params():
        return {
            "ksize": (1, 31, 5, 2),
            "sigmaX": (0.1, 10.0, 1.0, 0.1),
            "sigmaY": (0.0, 10.0, 0.0, 0.1),
            "borderType": list(BorderType),
        }


class MedianBlur(CvFunction):
    @staticmethod
    def process(image, ksize):
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def get_params():
        return {"ksize": (3, 31, 5, 2)}  # ksize must be odd and greater than 1


class BilateralFilter(CvFunction):
    @staticmethod
    def process(image, d, sigmaColor, sigmaSpace):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    @staticmethod
    def get_params():
        return {
            "d": (1, 100, 5, 1),  # Diameter of each pixel neighborhood
            "sigmaColor": (1.0, 100.0, 75.0, 1.0),  # Filter sigma in the color space
            "sigmaSpace": (
                1.0,
                100.0,
                75.0,
                1.0,
            ),  # Filter sigma in the coordinate space
        }
