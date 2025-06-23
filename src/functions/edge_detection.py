import cv2
from .enums import BorderType
from .function import CvFunction


class CannyEdgeDetection(CvFunction):
    @staticmethod
    def process(image, threshold1, threshold2, apertureSize=3, L2gradient=False):
        return cv2.Canny(
            image,
            threshold1,
            threshold2,
            apertureSize=apertureSize,
            L2gradient=L2gradient,
        )

    @staticmethod
    def get_params():
        return {
            "threshold1": (0, 10000, 100, 1),
            "threshold2": (0, 10000, 200, 1),
            "apertureSize": (3, 31, 3, 2),  # Should be 3, 5, or 7
            "L2gradient": False,
        }


class SobelEdgeDetection(CvFunction):
    @staticmethod
    def process(
        image, dx, dy, ksize=3, scale=1, delta=0, borderType=BorderType.BORDER_DEFAULT
    ):
        return cv2.Sobel(
            image,
            cv2.CV_8U,
            dx,
            dy,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=borderType.value,
        )

    @staticmethod
    def get_params():
        return {
            "dx": (1, 2, 1, 1),  # Order of the derivative x
            "dy": (1, 2, 0, 1),  # Order of the derivative y
            "ksize": (1, 31, 3, 2),  # Must be 1, 3, 5, or 7
            "scale": (1, 10, 1, 1),
            "delta": (0, 255, 0, 1),
            "borderType": list(BorderType),
        }


class Laplacian(CvFunction):
    @staticmethod
    def process(image, ksize=1, scale=1, delta=0, borderType=BorderType.BORDER_DEFAULT):
        return cv2.Laplacian(
            image,
            cv2.CV_8U,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=borderType.value,
        )

    @staticmethod
    def get_params():
        return {
            "ksize": (1, 31, 1, 2),  # Must be odd, or 1
            "scale": (1, 10, 1, 1),
            "delta": (0, 255, 0, 1),
            "borderType": list(BorderType),
        }
