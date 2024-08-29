import cv2
import numpy as np
from enum import Enum


class GaussianBlur:
    @staticmethod
    def process(image, ksize, sigmaX):
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX)

    @staticmethod
    def get_params():
        return {"ksize": (1, 31, 5, 2), "sigmaX": (0.1, 10.0, 1.0, 0.1)}


class CannyEdgeDetection:
    @staticmethod
    def process(image, threshold1, threshold2):
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def get_params():
        return {
            "threshold1": (0, 255, 100, 1),
            "threshold2": (0, 255, 200, 1),
        }


class ThresholdType(Enum):
    THRESH_BINARY = cv2.THRESH_BINARY
    THRESH_BINARY_INV = cv2.THRESH_BINARY_INV
    THRESH_TRUNC = cv2.THRESH_TRUNC
    THRESH_TOZERO = cv2.THRESH_TOZERO
    THRESH_TOZERO_INV = cv2.THRESH_TOZERO_INV


class Threshold:
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


class Dilate:
    @staticmethod
    def process(image, ksize, iterations):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def get_params():
        return {"ksize": (1, 31, 5, 2), "iterations": (1, 10, 1, 1)}


class Erode:
    @staticmethod
    def process(image, ksize, iterations):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def get_params():
        return {"ksize": (1, 31, 5, 2), "iterations": (1, 10, 1, 1)}
