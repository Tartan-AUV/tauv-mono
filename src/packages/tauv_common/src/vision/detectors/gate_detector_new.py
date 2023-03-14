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
from timeit import timeit
import time

import yaml
from skspatial.objects import Plane, Points, Line

from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from tauv_util.parms import Parms
import message_filters
import cv_bridge

import rospy
import tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GateDetector:
    GateCandidate = namedtuple('GateCandidate', 'l1 l2 lc side_alignment_score center_alignment_score depth_score')
    GatePosition = namedtuple('GatePosition', 'x y z q0 q1 q2 q3')

    PARMS_DEFAULT = Parms({
        'preprocessing': {
            'hsv_boundaries': [[[100, 0, 20], [170, 180, 80]]],
            'dilation_ks': 2,
            'erosion_ks': 5
        },
        'contours': {
            'min_size': 50,
            'min_hw_ratio': 5
        },
        'geometry': {
            'parallel_dth_max': 0.1,
            'side_endpoint_dev_f': 3,
            'side_alignment_score_cutoff': 0.5,
            'center_dev_cutoff': 0.2,
            'center_alignment_f': 3,
            'center_alignment_score_cutoff': 0.5,
        },
        'depth': {
            'sampling_interval': 10,
            'start_end_offset': 0,
            'outlier_f': 1.5,
            'dev_cutoff': 1.0,
            'dev_score_f': 2.0
        },
        'side_alignment_score_w': 1.0,
        'center_alignment_score_w': 1.0,
        'plane_dev_score_w': 1.0,
        'total_score_cutoff': 0.5
    })

    def __init__(self, parms: Parms = None):
        if parms:
            self.parms = parms
        else:
            self.parms = GateDetector.PARMS_DEFAULT

        self.__hfov = pi / 2
        self.__vfov = pi / 2
        self.__expected_tilt = 0.0

    def set_fov(self, hfov, vfov):
        self.__hfov = hfov
        self.__vfov = vfov

    def set_expected_tilt(self, tilt):
        self.__expected_tilt = tilt

    def detect(self, rgb: np.array, depth: np.array):
        p = self.parms
        candidates = self._detect_rgb(rgb)
        print(candidates)
        for gc in candidates:
            self.__get_position(depth, gc)
        detections = []
        for gc in candidates:
            r = self.__get_position(depth, gc)
            if r is None:
                continue
            pos, dev = r
            if dev > p.depth.dev_cutoff:
                continue
            dev_score = 1/(1+dev*p.depth.dev_score_f)
            norm_f = p.side_alignment_score_w + p.center_alignment_score_w + p.plane_dev_score_w
            score = (gc.side_alignment_score*p.side_alignment_score_w +
                     gc.center_alignment_score*p.center_alignment_score_w +
                     dev_score*p.plane_dev_score_w) / norm_f
            if score > p.total_score_cutoff:
                detections.append((pos, score))
        return detections

    def __get_position(self, depth_map: np.ndarray, gc: GateCandidate):
        p = self.parms.depth
        h, w = depth_map.shape[:2]
        k_h = w // 2 / math.tan(self.__hfov / 2)
        k_v = h // 2 / math.tan(self.__vfov / 2)

        def get_3d_point(p):
            center = np.array([w // 2, h // 2])
            pc = np.append(p - center, 1)
            x, y = p[0], p[1]
            if (not 0 <= x < w) or (not 0 <= y < h):
                raise ValueError("Point not in the image")
            return np.array([1, pc[0] / k_h, pc[1] / k_v]) * depth_map[y, x]

        # get_3d_point = np.vectorize(_get_3d_point)

        # depth_map_normalized = (depth_map * 25).astype(np.uint8)
        # depth_map_rgb = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_HOT)
        def sample_line(l):
            v = l[2:] - l[:2]
            n = np.linalg.norm(v)
            u = v / n * p.sampling_interval
            k = np.array([0, 0])
            points = []
            for i in range(p.start_end_offset, int(n / p.sampling_interval)-p.start_end_offset):
                points.append(tuple(l[:2] + u * i))
                # cv2.circle(depth_map_rgb, points[-1][0], 3, (0,255,0), 1)
            return np.array(points).astype(np.int32)


        pts1 = sample_line(gc.l1)
        pts2 = sample_line(gc.l2)
        ptsc = sample_line(gc.lc)

        # cv2.imshow('depth', depth_map_rgb)
        # cv2.waitKey(1)
        #

        def map_to_3d(pts2d):
            pts3d = np.zeros((len(pts2d), 3))
            for i in range(len(pts2d)):  # vectorize
                cv2.circle(depth_map, pts2d[i], 5, (0,0,255), 1)
                x, y = pts2d[i]
                if (not 0 <= x < w) or (not 0 <= y < h):
                    continue
                pts3d[i] = get_3d_point(pts2d[i])
            return pts3d

        pts3d1 = map_to_3d(pts1)
        pts3d2 = map_to_3d(pts2)
        pts3dc = map_to_3d(ptsc)

        if not len(pts1) or not len(pts2) or not len(ptsc):
            return None
        pts3d = np.concatenate((pts3d1, pts3d2, pts3dc))
        # remove zeros
        index = np.where(pts3d[:,0] == 0.0)
        pts3d = np.delete(pts3d, index, axis = 0)
        pts3d_depth = pts3d[:,0]
        # remove outliers
        quantiles = np.quantile(pts3d_depth, [.25, .75])
        q1 = quantiles[0]
        q3 = quantiles[1]
        iqr = q3-q1
        lower = q1 - p.outlier_f*iqr
        upper = q3 + p.outlier_f*iqr
        index = np.where(np.logical_or(pts3d_depth<lower, upper<pts3d_depth))
        pts3d = np.delete(pts3d, index, axis=0)
        # fit plane
        try:
            plane = Plane.best_fit(pts3d)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
            # plane.plot_3d(ax, alpha=0.2)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()

            # get standard deviation
            dev = 0
            for point in pts3d:
                dev += plane.distance_point(point) ** 2
            dev = math.sqrt(dev)
            # calculate midpoint (bottom of the center pole) and orientation
            mid_line = Line((0, 0, 0), pts3dc[-1])
            pos = plane.intersect_line(mid_line)
            v1 = gc.l1[:2]-gc.l1[2:]
            v2 = gc.l2[:2]-gc.l2[2:]
            u1 = v1/np.linalg.norm(v1)
            u2 = v2/np.linalg.norm(v2)
            u = (u1 + u2) / 2
            up = plane.project_vector([0, u[0], u[1]])  # points up
            up_u = up / np.linalg.norm(up)
            z = np.array([0,0,1])
            [q0, q1, q2] = np.cross(z, up_u)
            q3 = np.arccos(np.clip(np.dot(up_u, z), -1.0, 1.0))
            return GateDetector.GatePosition(*pos, q0, q1, q2, q3), dev

        except ValueError:
            return None

    def _detect_rgb(self, img):
        p = self.parms
        filtered = self._get_color_filter_mask(img, p.preprocessing.hsv_boundaries)
        # filtered = self.__filter_by_contour_size(mask)
        ds = p.preprocessing.dilation_ks
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*ds+1, 2 * ds + 1), (ds, ds))
        dilated = cv2.dilate(filtered,dilation_kernel)
        im_floodfill = dilated.copy()
        h, w = filtered.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        floodfilled = dilated | im_floodfill_inv
        es = p.preprocessing.erosion_ks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*es+1, 2*es+1), (es,es))
        eroded = cv2.erode(floodfilled, kernel)
        # cv2.imshow("eroded", eroded)
        edges = cv2.Canny(eroded, 0, 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for i, c in enumerate(contours):
            (cx, cy), (w, h), th = cv2.minAreaRect(c)
            if w > h:
                th -= 90.0
            w, h = min(w, h), max(w, h)
            if w == 0 or h < p.contours.min_size or h/w < p.contours.min_hw_ratio:
                continue
            th = -math.radians(th)
            dx = math.sin(th)*h/2
            dy = math.cos(th)*h/2
            lines.append([[cx-dx, cy-dy, cx+dx, cy+dy]])
        print(f'Extracted {len(lines)} lines from {len(contours)} contours')
        lines = np.array(lines).astype(np.int32)
        if len(lines) == 0:
            return []
        lines_polar = self._lines_cartesian_to_polar(lines[:,0,:])
        sorted_i = np.argsort(lines_polar[:, 1])
        parallel_ranges = self._get_continuous_ranges(lines_polar[sorted_i][:,1], p.geometry.parallel_dth_max)

        def show_pair(pair):
            img_cp = np.copy(img)
            l1 = lines[sorted_i][pair[0]]
            l2 = lines[sorted_i][pair[1]]
            GateDetector.draw_lines(img_cp, [l1, l2], 2, (0,255,0))
        def side_alignment_score(i1: int, i2: int):
            l1 = lines[sorted_i][i1, 0]
            l2 = lines[sorted_i][i2, 0]
            length1 = np.linalg.norm(l1[:2] - l1[2:])
            length2 = np.linalg.norm(l2[:2] - l2[2:])
            th1 = lines_polar[sorted_i][i1][1]
            rho1 = lines_polar[sorted_i][i1][0]
            rho2 = lines_polar[sorted_i][i2][0]
            drho = rho2 - rho1
            p11 = l1[:2]
            p21 = l2[:2]
            proj = p11 + [math.cos(th1) * drho, -math.sin(th1) * drho]
            ep_dev_px = np.linalg.norm(p21 - proj)
            ep_dev = ep_dev_px / length2 * p.geometry.side_endpoint_dev_f + 1
            return 1/ep_dev

        pairs = []
        for line in lines:
            cv2.line(img, line[0][:2], line[0][2:], (0, 255, 0), 2)
        for parallel_range_index, (start, end) in enumerate(parallel_ranges):
            for i1, i2 in itertools.combinations(range(start, end), 2):
                score = side_alignment_score(i1, i2)
                if score > p.geometry.side_alignment_score_cutoff:
                    pairs.append((i1, i2, parallel_range_index, score))
                    cv2.line(img, lines[sorted_i][i1,0][:2], lines[sorted_i][i1,0][2:], (0,0,
                                                                                        255), 2)
                    cv2.line(img, lines[sorted_i][i2,0][:2], lines[sorted_i][i2,0][2:], (0,0,
                                                                                         255), 2)

        print(f'Found {len(pairs)} pairs')

        pairs_with_center = []  # [index_l1, index_l2, index_c, side_alignment_score]
        for i1, i2, parallel_range_index, side_alignment_score in pairs:
            l1 = lines_polar[sorted_i][i1]
            l2 = lines_polar[sorted_i][i2]
            rho1, th1 = l1[0], l1[1]
            rho2, th2 = l2[0], l2[1]
            mid = (rho1+rho2)/2
            drho = p.geometry.center_dev_cutoff * abs(rho1-rho2)
            for i in range(*parallel_ranges[parallel_range_index]):
                lc = lines_polar[sorted_i][i]
                rhoc = lc[0]
                if mid-drho < rhoc < mid+drho:
                    pairs_with_center.append((i1, i2, i, side_alignment_score))

        print(f'Found {len(pairs_with_center)} pairs with center')
        candidates = []
        for i1, i2, ic, side_alignment_score in pairs_with_center:
            l1 = lines[sorted_i][i1][0]
            l2 = lines[sorted_i][i2][0]
            lc = lines[sorted_i][ic][0]
            th = (lines_polar[sorted_i][i1][1] + lines_polar[sorted_i][i2][1])/2
            side_length = (np.linalg.norm(l1[:2] - l1[2:]) + np.linalg.norm(l2[:2] - l2[2:]))/2
            u = np.array([-math.sin(th), -math.cos(th)])
            bottom_center = (l1[2:] + l2[2:]) / 2
            exp_bottom = bottom_center + u*side_length/410*325
            exp_top = bottom_center + u*side_length/410*510
            dist_sum = (np.linalg.norm(exp_top-lc[:2]) + np.linalg.norm(exp_bottom-lc[2:]))
            center_alignment_score = 1 / (dist_sum / side_length * 1.52 * p.geometry.center_alignment_f + 1)
            if center_alignment_score > p.geometry.center_alignment_score_cutoff:
                candidates.append(GateDetector.GateCandidate(l1, l2, lc, side_alignment_score, center_alignment_score, 0))

        print(f'Found {len(candidates)} candidates')
        return candidates

    @staticmethod
    def draw_lines(img, lines, width, color):
        if lines is None:
            return
        for line in lines:
            line = line[0]
            if len(line) == 4:
                cv2.line(img, line[0:2], line[2:4], color, width)
            else:
                raise NotImplementedError()

    @staticmethod
    def _get_color_filter_mask(img, hsv_boundaries):
        """
        Set all pixels with hsv values outside the given range to zero.
        :param img: image in RGB
        """
        hsv_boundaries_arr = np.array(hsv_boundaries, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        for boundary in hsv_boundaries_arr:
            new_mask = cv2.inRange(hsv, boundary[0], boundary[1])
            mask = cv2.bitwise_or(mask, new_mask)
        return mask

    @staticmethod
    def _get_continuous_ranges(arr, min_gap):
        # requires that arr is sorted, n x 1 array supporting comparison
        diff = np.diff(arr)
        gaps = np.where(diff >= min_gap)[0]
        gaps += 1
        ranges = np.empty((len(gaps) + 1, 2), dtype=np.int64)
        ranges[0, 0] = 0
        ranges[-1, 1] = len(arr)
        ranges[1:, 0] = gaps
        ranges[:-1, 1] = gaps
        return ranges

    @staticmethod
    def _lines_cartesian_to_polar(l):
        # y pointing down, theta from +x-axis ccw
        dx = l[:, 2] - l[:, 0]
        dy = l[:, 3] - l[:, 1]
        dx = np.where(dy < 0, -dx, dx)
        dy = np.abs(dy)
        th = np.arctan2(dx, dy)
        rho = (l[:, 0] - l[:, 1] * np.tan(th)) * np.cos(th)
        return np.transpose(np.array([rho, th]))


class GateDetectorNode(GateDetector):
    def __init__(self):
        print("Initializing Gate Detector...")
        super().__init__(Parms(rospy.get_param('~gate_detector_parameters')))
        self.rgb_sub = message_filters.Subscriber('oakd_front/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('oakd_front/stereo/depth_map',
                                                    Image)
        self.camera_info_sub = message_filters.Subscriber(
            'oakd_front/stereo/left/camera_info', CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub,
                                                    self.depth_sub,
                                                    self.camera_info_sub], 10)
        self.ts.registerCallback(self.callback)
        self.detection_pub = rospy.Publisher('gate_detections', PoseArray, queue_size=10)
        self.bridge = cv_bridge.CvBridge()
        print("Done.")

    def callback(self, rgb: Image, depth: Image, camera_info: CameraInfo):
        # Update FOV
        print('Callback')
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        w = camera_info.width
        h = camera_info.height
        hfov = 2*math.atan(w / (2*fx))
        vfov = 2*math.atan(h / (2*fy))
        self.set_fov(hfov, vfov)
        print(hfov, vfov)

        # Run detection
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        detections = self.detect(cv_rgb, cv_depth)
        # cv2.imshow('rgb', cv_rgb)
        # cv2.waitKey(1)
        pose_array = PoseArray()
        pose_array.header.stamp = rgb.header.stamp
        pose_array.header.frame_id = "kf/vehicle"
        for pos, score in detections:
            pose = Pose()
            pose.position = Point(pos.x, pos.y, pos.z)
            pose.orientation = Quaternion(pos.q0, pos.q1, pos.q2, pos.q3)
            pose_array.poses.append(pose)

        self.detection_pub.publish(pose_array)

def main():
    rospy.init_node('gate_detector', anonymous=True)
    g = GateDetectorNode()
    rospy.spin()

main()


# if __name__ == "__main__":
#     g = GateDetector()
#     p = "C:/Users/Gleb Ryabtsev/CMU/TAUV/TAUV-Assets/gate_sim_close_up.png"
#     v = "C:/Users/Gleb Ryabtsev/CMU/TAUV/TAUV-Assets/gate_gopro_full.mp4"
#     # img = cv2.imread(p, cv2.IMREAD_COLOR)
#     # g.detect(img, None)
#     cap = cv2.VideoCapture(v)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             g.detect(frame, None)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()

