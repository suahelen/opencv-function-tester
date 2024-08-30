import cv2
import numpy as np
from .enums import *
from .function import CvFunction


class BlobDetector(CvFunction):
    @staticmethod
    def process(
        image,
        minThreshold,
        maxThreshold,
        minArea,
        minCircularity,
        minConvexity,
        minInertiaRatio,
    ):
        # Set up the detector with default parameters.
        params = cv2.SimpleBlobDetector.Params()

        params.minThreshold = minThreshold
        params.maxThreshold = maxThreshold
        params.filterByArea = True
        params.minArea = minArea
        params.filterByCircularity = True
        params.minCircularity = minCircularity
        params.filterByConvexity = True
        params.minConvexity = minConvexity
        params.filterByInertia = True
        params.minInertiaRatio = minInertiaRatio
        detector = cv2.SimpleBlobDetector.create(params)

        keypoints = detector.detect(image)

        # Draw detected blobs as red circles.
        return cv2.drawKeypoints(
            image,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

    @staticmethod
    def get_params():
        return {
            "minThreshold": (0, 255, 50, 1),
            "maxThreshold": (0, 255, 200, 1),
            "minArea": (0.0, 5000.0, 30.0, 1.0),
            "minCircularity": (0.0, 1.0, 0.1, 0.01),
            "minConvexity": (0.0, 1.0, 0.87, 0.01),
            "minInertiaRatio": (0.0, 1.0, 0.01, 0.01),
        }


class FastFeatureDetector(CvFunction):
    @staticmethod
    def process(image, threshold=10, nonmaxSuppression=True, type=2):
        fast = cv2.FastFeatureDetector.create(
            threshold=threshold, nonmaxSuppression=nonmaxSuppression, type=type
        )
        keypoints = fast.detect(image, None)
        return cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))

    @staticmethod
    def get_params():
        return {
            "threshold": (1, 100, 10, 1),
            "nonmaxSuppression": True,
            "type": list(FastFeatureType),
        }


class OrbFeatureDetector(CvFunction):
    @staticmethod
    def process(
        image,
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=OrbHarrisScore.HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    ):
        orb = cv2.ORB.create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType.value,
            patchSize=patchSize,
            fastThreshold=fastThreshold,
        )
        keypoints, _ = orb.detectAndCompute(image, None)
        return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    @staticmethod
    def get_params():
        return {
            "nfeatures": (500, 10000, 1000, 100),
            "scaleFactor": (1.1, 2.0, 1.2, 0.1),
            "nlevels": (3, 15, 8, 1),
            "edgeThreshold": (1, 50, 31, 1),
            "firstLevel": (0, 10, 0, 1),
            "WTA_K": (2, 4, 2, 1),
            "scoreType": list(OrbHarrisScore),
            "patchSize": (1, 50, 31, 2),
            "fastThreshold": (1, 50, 20, 1),
        }


class SiftFeatureDetector(CvFunction):
    @staticmethod
    def process(
        image,
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    ):
        sift = cv2.SIFT.create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )
        keypoints = sift.detect(image, None)
        return cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255))

    @staticmethod
    def get_params():
        return {
            "nfeatures": (0, 5000, 0, 100),
            "nOctaveLayers": (1, 10, 3, 1),
            "contrastThreshold": (0.01, 0.5, 0.04, 0.01),
            "edgeThreshold": (1, 50, 10, 1),
            "sigma": (0.1, 5.0, 1.6, 0.1),
        }


class AkazeFeatureDetector(CvFunction):
    @staticmethod
    def process(
        image,
        descriptor_type=AkazeDescriptorType.DESCRIPTOR_MLDB_UPRIGHT,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=0.001,
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=DiffusivityType.DIFF_PM_G2,
    ):
        akaze = cv2.AKAZE.create(
            descriptor_type=descriptor_type.value,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=nOctaves,
            nOctaveLayers=nOctaveLayers,
            diffusivity=diffusivity.value,
        )
        keypoints = akaze.detect(image, None)
        return cv2.drawKeypoints(image, keypoints, None, color=(255, 255, 0))

    @staticmethod
    def get_params():
        return {
            "descriptor_type": list(AkazeDescriptorType),
            "descriptor_size": (0, 64, 0, 1),
            "descriptor_channels": (1, 3, 3, 1),
            "threshold": (0.0001, 0.01, 0.001, 0.0001),
            "nOctaves": (1, 10, 4, 1),
            "nOctaveLayers": (1, 10, 4, 1),
            "diffusivity": list(DiffusivityType),
        }
