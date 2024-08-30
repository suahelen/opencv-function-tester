import cv2
import numpy as np
from .enums import *
from .function import CvFunction


class HistogramEqualization(CvFunction):
    @staticmethod
    def process(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def get_params():
        return {
            "clip_limit": (0.1, 10.0, 2.0, 0.1),
            "tile_grid_size": [(2, 2), (4, 4), (8, 8), (16, 16)],
        }


class ColorSpaceConversion(CvFunction):
    @staticmethod
    def process(image, conversion_code=ColorSpaceConversionType.COLOR_BGR2GRAY):
        return cv2.cvtColor(image, conversion_code.value)

    @staticmethod
    def get_params():
        return {
            "conversion_code": list(ColorSpaceConversionType),
        }
