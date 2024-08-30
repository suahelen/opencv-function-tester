import cv2
import numpy as np
from .function import CvFunction


class OpticalFlow(CvFunction):
    @staticmethod
    def process(
        prev_image,
        next_image,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
    ):
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def get_params():
        return {
            "pyr_scale": (0.1, 0.9, 0.5, 0.1),
            "levels": (1, 10, 3, 1),
            "winsize": (5, 30, 15, 2),
            "iterations": (1, 10, 3, 1),
            "poly_n": (3, 7, 5, 2),
            "poly_sigma": (0.1, 2.0, 1.2, 0.1),
        }
