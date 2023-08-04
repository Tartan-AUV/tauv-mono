import cv2
import numpy as np

from ..shape_detector.adaptive_color_thresholding import get_adaptive_color_thresholding, GetAdaptiveColorThresholdingParams


def filter(img):
    params = GetAdaptiveColorThresholdingParams(
        global_thresholds=[[0,0,0,255,255,255]],
        local_thresholds=[[-255,-80,255,-5]],
        window_size=35
    )

    act = get_adaptive_color_thresholding(img, params)

    cv2.imshow('act', act)



def detect_gate(img, depth):
    filter(img)

