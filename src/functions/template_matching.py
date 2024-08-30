import cv2
import numpy as np
import streamlit as st
from .enums import *
from .function import CvFunction
from csem_template_matcher.MatcherApi.Matcher import Matcher, MatcherSettings


class TemplateMatching(CvFunction):
    @staticmethod
    def process(image, template, method=TemplateMatchingMethod.CCOEFF_NORMED):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template_gray, method.value)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method.value in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (
            top_left[0] + template.shape[1],
            top_left[1] + template.shape[0],
        )
        cv2.rectangle(image, top_left, bottom_right, 255, 2)
        return image

    @staticmethod
    def get_params():
        return {
            "method": list(TemplateMatchingMethod),
            "requires_secondary_image": True,
        }


class FastMatcher(CvFunction):
    @staticmethod
    def process(image, template, **params):

        settings = MatcherSettings()
        for key, value in params.items():
            setattr(settings, key, value)

        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        matcher = Matcher(template)
        matcher.set_settings(settings)

        matcher.match(image)
        res = matcher.get_result()

        output = Matcher.draw_matches(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), res)

        st.write(f"Found {len(res)} matches")

        return output

    @staticmethod
    def get_params():

        return {
            "template": None,
            "m_ckSIMD": True,
            "m_iMaxPos": (1, 100, 50, 1),
            "m_dMaxOverlap": (0.0, 1.0, 0.0, 0.1),
            "m_dScore": (0.0, 1.0, 0.5, 0.01),
            "m_dToleranceAngle": (0, 360, 180, 1),
            "m_iMinReduceArea": (1, 1024, 256, 1),
            "m_bToleranceRange": False,
            "m_dTolerance1": (0, 360, 1, 1),
            "m_dTolerance2": (0, 360, 1, 1),
            "m_dTolerance3": (0, 360, 1, 1),
            "m_dTolerance4": (0, 360, 1, 1),
            "m_useSubPixel": False,
            "m_bStopLayer": True,
            "requires_secondary_image": True,
        }
