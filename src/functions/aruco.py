import cv2
import numpy as np
from enum import Enum


class ArucoDetector:
    @staticmethod
    def process(image, dictionary_type):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type.value)
        detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        return image

    @staticmethod
    def get_params():
        return {
            "dictionary_type": list(DictType),
        }


class CharucoBoardDetector:
    @staticmethod
    def process(
        image,
        dictionary_type,
        patternsize_x,
        patternsize_y,
        square_length,
        marker_length,
        **params
    ):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type.value)
        board = cv2.aruco.CharucoBoard(
            (patternsize_x, patternsize_y), square_length, marker_length, aruco_dict
        )

        detector_parameter = cv2.aruco.DetectorParameters()
        for key, value in params.items():
            # if value is Enum() take value
            if isinstance(value, Enum):
                value = value.value
            setattr(detector_parameter, key, value)

        detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_parameter)

        corners, corner_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if marker_corners is not None and len(marker_corners) > 0:
            image = cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

        if corners is not None and corner_ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                image, corners, corner_ids, (0, 255, 0)
            )
        return image

    @staticmethod
    def get_params():
        return {
            "dictionary_type": list(DictType),
            "patternsize_x": (1, 30, 16, 1),
            "patternsize_y": (1, 30, 22, 1),
            "square_length": (0.01, 1.0, 0.05, 0.01),
            "marker_length": (0.01, 1.0, 0.04, 0.01),
            "adaptiveThreshWinSizeMin": (3, 50, 3, 1),
            "adaptiveThreshWinSizeMax": (3, 100, 23, 1),
            "adaptiveThreshWinSizeStep": (1, 50, 10, 1),
            "adaptiveThreshConstant": (1.0, 100.0, 7.0, 0.1),
            "minMarkerPerimeterRate": (0.01, 0.5, 0.03, 0.01),
            "maxMarkerPerimeterRate": (1.0, 10.0, 4.0, 0.1),
            "polygonalApproxAccuracyRate": (0.01, 0.1, 0.03, 0.01),
            "minCornerDistanceRate": (0.01, 0.1, 0.05, 0.01),
            "minDistanceToBorder": (1, 10, 3, 1),
            "minMarkerDistanceRate": (0.01, 0.1, 0.05, 0.01),
            "minGroupDistance": (0.0, 1.0, 0.0, 0.01),
            "cornerRefinementMethod": list(CornerRefineMethod),
            "cornerRefinementWinSize": (1, 10, 5, 1),
            "relativeCornerRefinmentWinSize": (0.01, 0.1, 0.01, 0.01),
            "cornerRefinementMaxIterations": (1, 100, 30, 1),
            "cornerRefinementMinAccuracy": (0.01, 1.0, 0.1, 0.01),
            "markerBorderBits": (1, 5, 1, 1),
            "perspectiveRemovePixelPerCell": (1, 50, 8, 1),
            "perspectiveRemoveIgnoredMarginPerCell": (0.0, 1.0, 0.13, 0.01),
            "maxErroneousBitsInBorderRate": (0.0, 1.0, 0.35, 0.01),
            "minOtsuStdDev": (0.0, 10.0, 5.0, 0.1),
            "errorCorrectionRate": (0.0, 1.0, 0.6, 0.1),
            "aprilTagQuadDecimate": (0.0, 1.0, 0.0, 0.1),
            "aprilTagQuadSigma": (0.0, 1.0, 0.0, 0.1),
            "aprilTagMinClusterPixels": (1, 100, 5, 1),
            "aprilTagMaxNmaxima": (1, 100, 10, 1),
            "aprilTagCriticalRad": (0.0, 50.0, 10.0, 0.1),
            "aprilTagMaxLineFitMse": (0.0, 50.0, 10.0, 0.1),
            "aprilTagMinWhiteBlackDiff": (1, 100, 5, 1),
            "aprilTagDeglitch": (0, 1, 0, 1),
            "detectInvertedMarker": False,
            "useAruco3Detection": False,
            "minSideLengthCanonicalImg": (1, 100, 16, 1),
            "minMarkerLengthRatioOriginalImg": (0.0, 1.0, 0.01, 0.01),
        }


class CornerRefineMethod(Enum):
    CORNER_REFINE_NONE = cv2.aruco.CORNER_REFINE_NONE
    CORNER_REFINE_SUBPIX = cv2.aruco.CORNER_REFINE_SUBPIX
    CORNER_REFINE_CONTOUR = cv2.aruco.CORNER_REFINE_CONTOUR


class DictType(Enum):
    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv2.aruco.DICT_7X7_50
    DICT_7X7_100 = cv2.aruco.DICT_7X7_100
    DICT_7X7_250 = cv2.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000
    DICT_ARUCO_ORIGINAL = cv2.aruco.DICT_ARUCO_ORIGINAL
    DICT_APRILTAG_16h5 = cv2.aruco.DICT_APRILTAG_16h5
    DICT_APRILTAG_16H5 = cv2.aruco.DICT_APRILTAG_16H5
    DICT_APRILTAG_25h9 = cv2.aruco.DICT_APRILTAG_25h9
    DICT_APRILTAG_25H9 = cv2.aruco.DICT_APRILTAG_25H9
    DICT_APRILTAG_36h10 = cv2.aruco.DICT_APRILTAG_36h10
    DICT_APRILTAG_36H10 = cv2.aruco.DICT_APRILTAG_36H10
    DICT_APRILTAG_36h11 = cv2.aruco.DICT_APRILTAG_36h11
    DICT_APRILTAG_36H11 = cv2.aruco.DICT_APRILTAG_36H11
    DICT_ARUCO_MIP_36h12 = cv2.aruco.DICT_ARUCO_MIP_36h12
    DICT_ARUCO_MIP_36H12 = cv2.aruco.DICT_ARUCO_MIP_36H12
