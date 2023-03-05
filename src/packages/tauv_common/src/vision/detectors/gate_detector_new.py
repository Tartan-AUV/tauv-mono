#!/usr/bin/env python3

# Author: Gleb Ryabtsev, 2023

import copy
import inspect
import itertools
import math
import sys
from functools import wraps
from inspect import signature
import cv2
import numpy as np
from math import sin, cos, pi, isnan, isinf
from typing import List, Tuple, OrderedDict
from collections import namedtuple

import yaml
from skspatial.objects import Plane, Points, Line

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
from tauv_util.parms import Parms
import message_filters
import cv_bridge

import rospy

class GateDetector:
    GateCandidate = namedtuple('GateCandidate', 'p0 p1 v')
    GatePosition = namedtuple('GatePosition', 'x y z roll pitch yaw dev')

    PARMS_DEFAULT = Parms({
        'filtering': {
            'hsv_boundaries': [[[100, 0, 0], [190, 255, 100]]],
            'min_contour_size': 50
        },
        'candidate_search': {
            'ht': {
                'adaptive': True,
                'adjustment_factor': 0.85,
                'min_lines': 20,
                'max_lines': 200,
                'blur': 5,
                'theta_steps': 90,
                'threshold': 200
            },
            'clustering_drho_min': 30.0,
            'clustering_dtheta_min': 0.2,
            'parallel_dtheta_max': 0.1,
        },
        'candidate_filtering': {
            'min_gate_size': 0.3,
            'max_tilt': 0.25,
            'gate_w': 3.048,
            'center_h': 0.609,
            'gate_pipe_w': 0.075,
            'side_h': 1.5,
            'depth_sampling_interval': 0.2,
            'center_score_cutoff': 0.5,
            'depth_std_dev_cutoff': 0.5,
            'score_coefficients': [1, 1]
        }
    })

    def __init__(self, parms: Parms):
        if parms:
            self.parms = parms
        else:
            self.parms = GateDetector.PARMS_DEFAULT

        self.ht_threshold = self.parms.candidate_search.ht.threshold
        self.__hfov = pi/2
        self.__vfov = pi/2
        self.__expected_tilt = 0.0

    def set_fov(self, hfov, vfov):
        self.__hfov = hfov
        self.__vfov = vfov

    def set_expected_tilt(self, tilt):
        self.__expected_tilt = tilt

    def detect(self, rgb: np.array, depth: np.array):
        p = self.parms.candidate_filtering
        assert(rgb.shape == depth.shape)
        candidates = self.__find_candidates(rgb)
        if not candidates:
            return None
        # depth_std_dev_cutoff = self.__params['candidate_filtering']['depth_std_dev_cutoff']
        # center_score_cutoff = self.__params['candidate_filtering']['center_score_cutoff']
        # coeffs = self.__params['candidate_filtering']['score_coefficients']
        scores = []
        positions = []
        for gc in candidates:
            pos = self.__get_position(depth, gc)
            if pos.dev > p.depth_std_dev_cutoff:
                continue
            center_score = self.__get_center_score(rgb, gc)
            if center_score > p.center_score_cutoff:
                continue
            positions.append(pos)
            scores.append(center_score*p.coeffs[0]+pos.dev*p.coeffs[1])

        index = max(range(len(scores)), key=lambda i: scores[i])
        return positions[index]

    def __get_position(self, depth_map: np.ndarray, gc: GateCandidate) -> GatePosition:
        w, h = depth_map.shape
        k_h = w//2 / math.tan(self.__hfov / 2)
        k_v = h//2 / math.tan(self.__vfov / 2)
        p = self.parms.candidate_filtering
        # side_h = self.__params['candidate_filtering']['side_h']
        # center_h = self.__params['candidate_filtering']['center_h']
        # gate_w = self.__params['candidate_filtering']['gate_w']
        # d = self.__params['candidate_filtering']['depth_sampling_interval']
        projection_mat = np.array([[0, 0, 1,
                           1/k_h, 0, 0,
                           0, 1/k_v, 0]])

        def _get_3d_point(p):
            center = np.array([w//2, h//2])
            pc = np.append(p - center, 1)
            return np.array([1, pc[0]/k_h, pc[1]/k_v])*depth_map[p[0], p[1]]
        get_3d_point = np.vectorize(_get_3d_point)

        def sample_line(pt1, pt2):
            v = pt2 - pt1
            n = np.linalg.norm(v)
            u = v / n * p.depth_sampling_interval
            k = np.array([0,0])
            points = np.array((n / p.depth_sampling_interval, 2), dtype=np.int32)
            for i in range(len(points)):
                points[i] = pt1 + u*i
            return points

        # gc.p0-----gc.p1
        # |      |      |
        # |      p3     |
        # p2           p4

        p0, p1 = gc.p0, gc.p1
        pixel_w = np.linalg.norm(p1 - p0)
        scale = pixel_w / p.gate_w
        p2 = p0 + gc.v * int(p.side_h*scale)
        mid = (p0 + p1)//2
        p3 = mid + gc.v * int(p.center_h*scale)
        p4 = p1 * gc.v * int(p.side_h*scale)

        pts02 = sample_line(p0, p2, )
        pts01 = sample_line(p0, p1)
        ptsmid3 = sample_line(mid, p3)
        pts14 = sample_line(p1, p4)
        pts2d = np.concatenate((pts02, pts01, ptsmid3, pts14))

        pts3d = get_3d_point(pts2d)

        plane = Plane.best_fit(pts3d)

        dev = 0
        for point in pts3d:
            dev += plane.distance_point(point)**2
        dev = math.sqrt(dev)

        mid_line = Line((0,0,0), get_3d_point(mid))
        pos = plane.intersect_line(mid_line)
        up = plane.project_vector([0, -gc.v[0], -gc.v[1]])  # points up
        right_2d = p0-p1
        if right_2d[0] < 0:
            right_2d *= -1
        right = plane.project_vector([0, right_2d[0], right_2d[1]])

        forward = np.cross(up, right)
        roll = math.atan2(right[1], np.dot(forward, right))
        pitch = math.atan2(forward[2], np.dot(right, forward))
        yaw = math.atan2(forward[1], forward[0])

        return GateDetector.GatePosition(*mid, roll, pitch, yaw)

    def __find_candidates(self, img):
        p = self.parms.candidate_filtering
        mask = self.__get_color_filter_mask(img)
        cv2.imshow("color_filtered", cv2.bitwise_and(img, img, mask=mask))
        filtered = self.__filter_by_contour_size(mask)
        lines = self.__find_lines(filtered)
        if lines is None:
            return None
        median_lines = self.__find_clusters(lines)
        parallel_groups = self.__find_parallel_groups(median_lines)
        gate_candidates = list()
        # max_tilt = self.__params['candidate_filtering']['max_tilt']
        for horizontal_candidate in median_lines:
            for vertical_candidate_group in parallel_groups:
                if len(vertical_candidate_group) == 1:
                    continue
                intersections = self.__intersect_group(horizontal_candidate, vertical_candidate_group)
                for i, j in itertools.combinations(range(len(intersections)), 2):
                    x1, y1, _, theta1 = intersections[i]
                    x2, y2, _, theta2 = intersections[j]
                    theta_mean = (theta1 + theta2) / 2
                    v = np.array([-sin(theta_mean), cos(theta_mean)])
                    if ((abs(self.__expected_tilt + theta_mean) > p.max_tilt) and
                            (180 - abs(self.__expected_tilt + theta_mean) > p.max_tilt)):
                        continue
                    gate_candidates.append(self.GateCandidate(np.array((x1, y1)),
                                                              np.array((x2, y2)),
                                                              v))
        gate_candidates_filtered = []
        min_gate_size = p.min_gate_size * img.shape[0]
        for gc in gate_candidates:
            v = gc.p1 - gc.p0
            if np.linalg.norm(v) < min_gate_size:
                continue
            gate_candidates_filtered.append(gc)

        if gate_candidates_filtered:
            print(len(gate_candidates_filtered))
            gc = max(gate_candidates_filtered, key=lambda gc: self.__get_center_score(filtered, gc))
        else:
            gc = None

        return gate_candidates_filtered

        # if lines is not None:
        #     for line in lines:
        #         rho, theta = line
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1500 * (-b))
        #         y1 = int(y0 + 1500 * (a))
        #         x2 = int(x0 - 1500 * (-b))
        #         y2 = int(y0 - 1500 * (a))
        #         cv2.line(img, (x1, y1), (x2, y2), (0, 100, 0), 1)
        # if gc:
        #     cv2.circle(img, gc.p0.astype(np.int32), 10, (255, 0, 0), -1)
        #     cv2.circle(img, gc.p1.astype(np.int32), 10, (255, 0, 0), -1)
        #     mid = gc.p0 + (gc.p1 - gc.p0) / 2
        #     cv2.circle(img, mid.astype(np.int32), 10, (0, 255, 0), -1)
        #     p0 = (mid-1500*gc.v).astype(int)
        #     p1 = (mid+1500*gc.v).astype(int)
        #     cv2.line(img, p0, p1, (0, 255, 0), 4)
        # cv2.imshow("img", img)

    def __get_color_filter_mask(self, img):
        """
        Set all pixels with hsv values outside the given range to zero.
        :param img: image in RGB
        """
        hsv_boundaries = np.array(self.parms.filtering.hsv_boundaries, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, _ = img.shape
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        cv2.imshow("test", img)
        for boundary in hsv_boundaries:
            new_mask = cv2.inRange(hsv, boundary[0], boundary[1])
            mask = cv2.bitwise_or(mask, new_mask)
        cv2.imshow("mask", mask)
        return mask

    def __filter_by_contour_size(self, img):
        p = self.parms.filtering
        # min_contour_size = self.__params['filtering']['min_contour_size']
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for c in contours:
            _, _, w, h = cv2.boundingRect(c)
            if w > p.min_contour_size or h > p.min_contour_size:
                filtered_contours.append(c)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        return mask

    def __find_lines(self, img):
        p = self.parms.candidate_search.ht
        # blur = self.__params['candidate_search']['ht']['blur']
        # theta_steps = self.__params['candidate_search']['ht']['theta_steps']
        # adaptive = self.__params['candidate_search']['ht']['adaptive']
        # min_lines = self.__params['candidate_search']['ht']['min_lines']
        # max_lines = self.__params['candidate_search']['ht']['max_lines']
        # adjustment_factor = self.__params['candidate_search']['ht']['adjustment_factor']
        image_blurred = cv2.GaussianBlur(img, (blur, blur), 0)
        lines = []
        counter = 0
        while not p.adaptive or not p.min_lines <= len(lines) <= p.max_lines:
            lines = cv2.HoughLines(image_blurred, 1, np.pi / p.theta_steps, self.ht_threshold)
            if len(lines) < p.min_lines:
                self.ht_threshold = int(self.ht_threshold * p.adjustment_factor)
            elif len(lines) > p.max_lines:
                self.ht_threshold = int(self.ht_threshold / p.adjustment_factor)
        print(f'Ctr: {counter}, thresh: {self.ht_threshold}, lines: {len(lines)}')
        if lines is not None:
            lines = lines[:, 0, :]
        return lines

    @staticmethod
    def __get_continuous_ranges(arr, gap_threshold, sort_by):
        arr_sorted = arr[np.argsort(arr[:, sort_by])]
        diff = np.diff(arr_sorted[:, sort_by])
        gap_indices = np.where(diff > gap_threshold)[0] + 1
        ranges = np.empty((len(gap_indices) + 1, 2), dtype=np.int64)
        if not len(gap_indices):
            ranges[0] = np.array([0, len(arr_sorted)])
            return arr_sorted, ranges

        prev_gap_index = 0
        for i in range(len(gap_indices)):
            ranges[i][0] = prev_gap_index
            ranges[i][1] = gap_indices[i]
            prev_gap_index = gap_indices[i]
        ranges[-1][0] = gap_indices[-1]
        ranges[-1][1] = len(arr_sorted)
        return arr_sorted, ranges

    def __find_clusters(self, lines):
        # clustering_dtheta_min = self.__params['candidate_search']['clustering_dtheta_min']
        # clustering_drho_min = self.__params['candidate_search']['clustering_drho_min']
        p = self.parms.candidate_search
        lines_theta_sorted, ranges_theta = self.__get_continuous_ranges(lines,
                                                                        p.clustering_dtheta_min, 1)

        median_lines = []

        for r_th in ranges_theta:
            rho_sorted, ranges_rho = self.__get_continuous_ranges(lines_theta_sorted[r_th[0]:r_th[1]],
                                                                  p.clustering_drho_min, 0)
            for r_rho in ranges_rho:
                start, end = r_rho[0], r_rho[1]
                median_index = (start + end) // 2
                if median_index < 0:
                    continue  # todo: this is dirty fix
                mean_theta = np.mean(rho_sorted[start:end, 1])
                median_line = np.array([rho_sorted[median_index][0], mean_theta])
                median_lines.append(median_line)

        return np.array(median_lines)

    def __find_parallel_groups(self, lines):
        # parallel_dtheta_max = self.__params['candidate_search']['parallel_dtheta_max']
        p = self.parms.candidate_search
        lines_sorted, ranges = self.__get_continuous_ranges(lines, p.parallel_dtheta_max, 1)
        result = []
        for i, (start, end) in enumerate(ranges):
            result.append(list())
            for j in range(start, end):
                result[i].append(list(lines_sorted[j]))
        return result

    def __intersect_group(self, line: Tuple[float, float], line_group: List[Tuple[float, float]]):
        """
        Calculate all intersections between a given line and lines in line_group
        :param line: Tuple (rho, theta)
        :param line_group: List of tuples (rho, theta)
        :return: List of tuples (x, y, rho, theta) representing the intersection points, each with
        the corresponding line from other_lines
        """
        rho1, theta1 = line
        intersections = []
        for group_line in line_group:
            intersection = self.__intersect_polar(line, group_line)
            if intersection is not None:
                intersections.append(intersection + tuple(group_line))
        return intersections

    def __intersect_polar(self, l1, l2):
        assert (-pi <= l1[1] <= pi)
        assert (-pi <= l2[1] <= pi)

        # Find the equation of each line
        m1 = np.tan(l1[1] + pi / 2)
        b1 = l1[0] / sin(l1[1])
        m2 = np.tan(l2[1] + pi / 2)
        b2 = l2[0] / sin(l2[1])

        # Find the intersection point of the two lines
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        if isnan(x) or isnan(y) or isinf(x) or isinf(y):
            return None
        return x, y

    def __get_center_score(self, img, gc: GateCandidate):
        # gate_w = self.__params['candidate_filtering']['gate_w']
        # center_h = self.__params['candidate_filtering']['center_h']
        # gate_pipe_w = self.__params['candidate_filtering']['gate_pipe_w']
        p = self.parms.candidate_filtering
        delta_hor = gc.p1 - gc.p0
        pts = np.zeros((4, 1, 2), np.int32)
        center_region_w_rel = p.gate_pipe_w / p.gate_w
        pts[0][0] = gc.p0 + (1 - center_region_w_rel) / 2 * delta_hor
        pts[3][0] = gc.p0 + (1 + center_region_w_rel) / 2 * delta_hor
        center_h_px = np.linalg.norm(delta_hor) / p.gate_w * p.center_h
        pts[1][0] = pts[0][0] + center_h_px * gc.v
        pts[2][0] = pts[3][0] + center_h_px * gc.v
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        mask_area = np.count_nonzero(mask)
        if mask_area > 0:
            score = np.count_nonzero(masked_img) / mask_area
        else:
            score = 0
        return score


class GateDetectorNode(GateDetector):

    def __init__(self):
        config_path = "../../../../tauv_config/kingfisher_sim_description/yaml/gate_detector.yaml"
        super().__init__(Parms.fromfile(config_path))
        self.rgb_sub = message_filters.Subscriber(
            '/kf/vehicle/oakd_front/stereo/left/image_rect', Image)
        self.depth_sub = message_filters.Subscriber('/kf/vehicle/oakd_front/stereo/depth_map',
                                                    Image)
        self.camera_info_sub = message_filters.Subscriber(
            '/kf/vehicle/oakd_front/stereo/left/camera_info', CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub,
                                                    self.depth_sub,
                                                    self.camera_info_sub], 10)
        self.ts.registerCallback(self.callback)
        self.detection_pub = rospy.Publisher('gate_detection_pub', Pose, queue_size=10)
        self.bridge = cv_bridge.CvBridge()

    def callback(self, rgb: Image, depth: Image, camera_info: CameraInfo):
        # Update FOV
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        w = camera_info.width
        h = camera_info.height
        hfov = 2*math.atan(w / (2*fx))
        vfov = 2*math.atan(h / (2*fy))
        self.set_fov(hfov, vfov)

        # Run detection
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        cv_depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        detection = self.detect(cv_rgb, cv_depth)
        print(detection)


def main():
    rospy.init_node('gate_detector', anonymous=True)
    g = GateDetectorNode()
    rospy.spin()


main()
