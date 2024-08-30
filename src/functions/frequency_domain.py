import cv2
import numpy as np
from .function import CvFunction


class FourierTransform(CvFunction):
    @staticmethod
    def process(image, dft_scale=1, dft_shift=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        if dft_shift:
            dft_shift = np.fft.fftshift(dft)
        else:
            dft_shift = dft
        magnitude_spectrum = dft_scale * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        )
        return cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

    @staticmethod
    def get_params():
        return {
            "dft_scale": (0.1, 10.0, 1.0, 0.1),
            "dft_shift": True,
        }


class InverseFourierTransform(CvFunction):
    @staticmethod
    def process(image):
        dft = cv2.idft(image)
        return cv2.magnitude(dft[:, :, 0], dft[:, :, 1])

    @staticmethod
    def get_params():
        return {}
