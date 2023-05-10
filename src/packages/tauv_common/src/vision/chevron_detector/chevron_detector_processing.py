import numpy as np
import cv2
from math import atan2, pi
from enum import IntEnum


def threshold(img, hsv_ranges):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w, _ = img_hsv.shape

    img_thresh = np.zeros((h, w), dtype=np.uint8)
    for hsv_range in hsv_ranges:
        low = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
        high = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
        mask = cv2.inRange(img_hsv, low, high)
        img_thresh = img_thresh | mask

    return img_thresh


def get_border_mask(img, left, right, top, bottom):
    mask = np.ones((img.shape[0], img.shape[1]))

    mask[:, :left] = 0
    mask[:, -right:] = 0
    mask[top:, :] = 0
    mask[-bottom:, :] = 0

    return mask


def clean(img, close_size, open_size):
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_open = np.ones((open_size, open_size), np.uint8)

    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open)

    return img_opened


def get_components(img, area_threshold):
    labels, n_labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    label = 0
    out_labels = np.zeros(labels.shape, dtype=np.uint8)

    for i in range(n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            out_labels[labels == i] = label

            label += 1

    return out_labels, label


def get_contour(img, approximation_factor):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]

    perimeter = cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, approximation_factor * perimeter, True)

    return contour_approx


def get_angles(contour):
    n_points = contour.shape[0]

    angles = np.zeros((n_points))

    for i in range(n_points):
        x1, y1 = contour[(i - 1) % n_points, :]
        x2, y2 = contour[i, :]
        x3, y3 = contour[(i + 1) % n_points, :]

        angle = atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2)

        angle = (angle + pi) % (2 * pi)

        if angle > pi:
            angle = angle - pi

        angles[i] = angle

    return angles


class AngleClassification(IntEnum):
    NONE = 0
    FRONT = 1
    BACK = 2
    SIDE = 3
    TAIL = 4


def classify_angles(contour, angles, angle_threshold, mask):
    n_points = contour.shape[0]

    hull = cv2.convexHull(contour, returnPoints=False)

    classifications = np.zeros((n_points))

    for i in range(n_points):
        classification = AngleClassification.NONE

        if mask[contour[i]] != 0:
            if np.abs(angles[i] - (pi / 2)) < angle_threshold:
                if contour[i] in hull:
                    classification = AngleClassification.FRONT
                else:
                    classification = AngleClassification.BACK
            elif np.abs(angles[i] - (3 * pi / 4)) < angle_threshold and contour[i] in hull:
                classification = AngleClassification.SIDE
            elif np.abs(angles[i] - (pi / 4)) < angle_threshold and contour[i] in hull:
                classification = AngleClassification.TAIL

        classifications[i] = classification.value

    return classifications