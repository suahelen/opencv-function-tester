from functions.blurring import GaussianBlur, MedianBlur, BilateralFilter
from functions.edge_detection import CannyEdgeDetection, SobelEdgeDetection, Laplacian
from functions.thresholding import Threshold
from functions.morphological import Dilate, Erode, Morphology
from functions.transformations import Resize
from functions.aruco import ArucoDetector, CharucoBoardDetector
from functions.feature_detection import (
    BlobDetector,
    FastFeatureDetector,
    OrbFeatureDetector,
    SiftFeatureDetector,
    AkazeFeatureDetector,
)
from functions.shape_detection import ContourDetection, HoughLines, HoughCircles
from functions.frequency_domain import FourierTransform, InverseFourierTransform
from functions.histogram import (
    HistogramEqualization,
    ColorSpaceConversion,
)

# from functions.optical_flow import OpticalFlow
from functions.template_matching import (
    TemplateMatching,
    FastMatcher,
)

opencv_functions = {
    "Gaussian Blur": GaussianBlur,
    "Canny Edge Detection": CannyEdgeDetection,
    "Threshold": Threshold,
    "Dilate": Dilate,
    "Erode": Erode,
    "Median Blur": MedianBlur,
    "Bilateral Filter": BilateralFilter,
    "Sobel Edge Detection": SobelEdgeDetection,
    "Laplacian": Laplacian,
    "Morphology": Morphology,
    "Resize": Resize,
    "Aruco Detector": ArucoDetector,
    "Charuco Board Detector": CharucoBoardDetector,
    "Blob Detector": BlobDetector,
    "Fast Feature Detector": FastFeatureDetector,
    "Orb Feature Detector": OrbFeatureDetector,
    "Sift Feature Detector": SiftFeatureDetector,
    "Akaze Feature Detector": AkazeFeatureDetector,
    "Contour Detection": ContourDetection,
    "Hough Lines": HoughLines,
    "Hough Circles": HoughCircles,
    "Fourier Transform": FourierTransform,
    "Inverse Fourier Transform": InverseFourierTransform,
    "Histogram Equalization": HistogramEqualization,
    "Color Space Conversion": ColorSpaceConversion,
    "Template Matching": TemplateMatching,
    "Fast Matcher": FastMatcher,
}
