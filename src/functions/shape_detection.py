import cv2
import numpy as np
from .function import CvFunction


class HoughCircles(CvFunction):
    @staticmethod
    def process(image, dp, minDist, param1, param2, minRadius, maxRadius):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        return image

    @staticmethod
    def get_params():
        return {
            "dp": (1.0, 2.0, 1.2, 0.1),
            "minDist": (1, 100, 20, 1),
            "param1": (50, 300, 100, 1),
            "param2": (30, 200, 100, 1),
            "minRadius": (0, 100, 0, 1),
            "maxRadius": (0, 100, 0, 1),
        }


class HoughLines(CvFunction):
    @staticmethod
    def process(image, rho, theta, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLines(gray, rho, theta, threshold)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image

    @staticmethod
    def get_params():
        return {
            "rho": (1, 10, 1, 1),
            "theta": (0.01, 1, 0.01, 0.01),
            "threshold": (1, 100, 50, 1),
        }


class ContourDetection(CvFunction):
    @staticmethod
    def process(image, retrieval_mode, approximation_method):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, retrieval_mode.value, approximation_method.value
        )
        return cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    @staticmethod
    def get_params():
        return {
            "retrieval_mode": list(
                cv2.RETR_TREE, cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP
            ),
            "approximation_method": list(
                cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE
            ),
        }
