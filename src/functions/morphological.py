import cv2
from .enums import MorphShape, MorphOperation
from .function import CvFunction


class Dilate(CvFunction):
    @staticmethod
    def process(image, ksize, iterations, shape=MorphShape.MORPH_RECT):
        kernel = cv2.getStructuringElement(shape.value, (ksize, ksize))
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def get_params():
        return {
            "ksize": (1, 31, 5, 2),
            "iterations": (1, 10, 1, 1),
            "shape": list(MorphShape),
        }


class Erode(CvFunction):
    @staticmethod
    def process(image, ksize, iterations, shape=MorphShape.MORPH_RECT):
        kernel = cv2.getStructuringElement(shape.value, (ksize, ksize))
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def get_params():
        return {
            "ksize": (1, 31, 5, 2),
            "iterations": (1, 10, 1, 1),
            "shape": list(MorphShape),
        }


class Morphology(CvFunction):
    @staticmethod
    def process(image, operation, ksize, shape=MorphShape.MORPH_RECT, iterations=1):
        kernel = cv2.getStructuringElement(shape.value, (ksize, ksize))
        return cv2.morphologyEx(image, operation.value, kernel, iterations=iterations)

    @staticmethod
    def get_params():
        return {
            "operation": list(MorphOperation),
            "ksize": (1, 31, 5, 2),
            "shape": list(MorphShape),
            "iterations": (1, 10, 1, 1),
        }
