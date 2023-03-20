#!/usr/bin/env python3
import time
import timeit
from typing import List

# Author: Gleb Ryabtsev, 2023

import cv2
import numpy as np
import cv_bridge
from skspatial.objects import Plane, Line
from scipy.spatial.transform import Rotation

import rospy
import message_filters
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo

import math
import itertools
from math import pi

from collections import namedtuple
from dataclasses import dataclass
from tauv_util.parms import Parms


class GateDetector:
    GateCandidate = namedtuple('GateCandidate',
                               'l1 l2 lc side_alignment_score center_alignment_score depth_score')
    GatePosition = namedtuple('GatePosition', 'x y z q0 q1 q2 q3')

    @dataclass
    class _GateCandidate:
        left_pole_img: np.ndarray
        right_pole_img: np.ndarray
        img_pos: np.ndarray
        img_rot: Rotation
        depth_pos: np.ndarray = None
        depth_rot: Rotation = None
        depth_dev: float = None

    def __init__(self, parms: Parms):
        self._parms = parms
        self._camera_matrix = np.array([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, 1]])
        self._camera_matrix_inv = self._inv_camera_matrix(self._camera_matrix)

    def set_camera_matrix(self, camera_matrix):
        """
        Set camera parameters
        @param camera_matrix: Intrinsics matrix (3x3)
        ( 4, 5, 8, 12 or 14 elements)
        @return:
        """
        self._camera_matrix = np.array(camera_matrix).reshape(3, 3)
        # get inverse
        self._camera_matrix_inv = self._inv_camera_matrix(self._camera_matrix)

    @staticmethod
    def _inv_camera_matrix(mat):
        """
        Calculate the inverse of a camera matrix
        @param mat: 3x3 np camera matrix
        @return: 3x4 inverted camera matrix
        """
        inverse_3x3 = np.linalg.inv(mat)
        inverse_3x4 = np.concatenate((inverse_3x3, [[0], [0], [0]]), axis=1)
        # inverse_4x4 = np.concatenate((inverse_3x4, [[0, 0, 0, 1]]), axis=0)
        return inverse_3x4

    def detect(self, color: np.array, depth: np.array):
        p = self._parms
        print()
        # find gate candidates on the rgb image
        candidates = self._find_candidates(color)

        # clip position and orientation
        candidates = self._clip_candidates(candidates)

        self._calculate_depth_poses(candidates, depth)

        candidates = self._filter_by_depth_dev(candidates)

        candidates = self._filter_by_pose_diff(candidates)
        print(f'{len(candidates)} after pose diff filtering')

        detections = []
        for c in candidates:
            d = self.GatePosition(*c.depth_pos, *c.depth_rot.as_quat())
            detections.append(d)
        return detections

    def _filter_by_depth_dev(self,
                             candidates: List[_GateCandidate]):
        """
        Filters a list of GateCandidates based on the deviation from the
        best fit plane
        @param candidates: List of GateCandidates
        @return: List of GateCandidates
        """
        p = self._parms

        result = []
        for c in candidates:
            if c.depth_dev < p.depth.dev_cutoff:
                result.append(c)

        return result

    def _calculate_depth_poses(self,
                               candidates: List[_GateCandidate],
                               depth_map: np.ndarray):
        """
        @param c: Array of gate candidates
        @param depth: Depth image (1 channel)
        @return: None
        """
        p = self._parms.depth

        for gc in candidates:
            # print("")
            # print("PROCESSING CANDIDATE")
            t = time.monotonic()
            # Sample lines from the depth map
            left_top, left_bottom = gc.left_pole_img[:2], gc.left_pole_img[2:]
            right_top, right_bottom = gc.right_pole_img[:2], gc.right_pole_img[2:]
            left_2d = self._get_line_array(left_top, left_bottom, depth_map)
            right_2d = self._get_line_array(right_top, right_bottom, depth_map)
            pts_2d = np.concatenate((left_2d, right_2d), axis=0)
            pts_2d = pts_2d[::p.sampling_interval]

            t1 = time.monotonic()
            # print(f"LS: {t1-t}")
            t=t1

            # Remove points with depth=0
            depth = pts_2d[:, 2]
            index = np.where(depth == 0.0)
            pts_2d = np.delete(pts_2d, index, axis=0)

            t1 = time.monotonic()
            # print(f"D0RM: {t1 - t}")
            t = t1

            # Remove outliers
            depth = pts_2d[:, 2]
            quantiles = np.quantile(depth, [.25, .75])
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower = q1 - p.outlier_f * iqr
            upper = q3 + p.outlier_f * iqr
            index = np.where(np.logical_or(depth < lower, upper < depth))
            pts_2d = np.delete(pts_2d, index, axis=0)

            t1 = time.monotonic()
            # print(f"OURM: {t1 - t}")
            t = t1

            # Prepare array of 2D points for matrix multiplication
            # pts_2d_4: [[u*depth, v*depth, depth, 1], ...]
            pts_2d_4 = np.zeros((len(pts_2d), 4), dtype=np.float64)
            pts_2d_4[:, 0] = pts_2d[:, 0] * pts_2d[:, 2]
            pts_2d_4[:, 1] = pts_2d[:, 1] * pts_2d[:, 2]
            pts_2d_4[:, 2] = pts_2d[:, 2]
            pts_2d_4[:, 3] = 1

            t1 = time.monotonic()
            # print(f"3DPP: {t1 - t}")
            t = t1

            # get 3D points (camera_matrix_inv x pts_2d_4)
            mat = self._camera_matrix_inv
            pts_3d = np.matmul(mat, pts_2d_4.transpose())
            pts_3d = pts_3d.transpose()

            t1 = time.monotonic()
            # print(f"G3DP: {t1 - t}")
            t = t1

            # fit a plane
            try:
                plane = Plane.best_fit(pts_3d)
            except ValueError:
                continue

            t1 = time.monotonic()
            # print(f"PF: {t1 - t}")
            t = t1

            # calculate deviation
            a, b, c, d = plane.cartesian()
            normal = np.array([a, b, c])
            normal_length = np.linalg.norm(normal)
            normal = normal / normal_length
            d /= normal_length
            plane_pt = normal * d * -1.0
            squared_distances = np.abs((pts_3d - plane_pt).dot(normal))
            gc.depth_dev = np.sum(squared_distances) / len(squared_distances)

            t1 = time.monotonic()
            # print(f"DEV: {t1 - t}")
            t = t1

            # calculate gate position based on the plane
            # get the gate center point vector
            center_2d = (left_top + left_bottom + right_top + right_bottom) / 4.0
            center_2d_4 = np.concatenate((center_2d, [1, 1]))
            center_3d_vec = np.matmul(mat, center_2d_4)[:3]
            center_3d_line = Line([0., 0., 0.], center_3d_vec)
            gc.depth_pos = np.array(plane.intersect_line(center_3d_line))

            t1 = time.monotonic()
            # print(f"POS: {t1 - t}")
            t = t1

            # calculate rotation
            if normal[2] < 0.0:
                normal *= (-1.0)
            z_unit = [0, 0, 1]
            x, y, z = np.cross(z_unit, normal)
            w = 1.0 + np.dot(normal, z_unit)
            gc.depth_rot = Rotation.from_quat((x, y, z, w))

            t1 = time.monotonic()
            # print(f"ROT: {t1 - t}")
            t = t1

    def _filter_by_pose_diff(self, candidates):
        """
        Filters candidates list by the difference between poses calcualted from
        the depth and the color images
        @param candidates: List of GateCandidates
        @return: List of GateCandidates
        """
        p = self._parms

        # Prepare thresholds (faster to compare squares)
        max_translation_diff_sq = p.depth.max_translation_diff ** 2
        max_rotation_diff_sq = math.radians(p.depth.max_rotation_diff) ** 2

        # Squared norm function
        sq_euclidean = lambda x: np.inner(x, x)

        result = []
        for c in candidates:
            if c.depth_pos is None or c.depth_rot is None:
                continue
            translation_diff_sq = sq_euclidean(c.img_pos - c.depth_pos)
            rotation_diff = c.depth_rot.inv()*c.img_rot
            angle_sq = sq_euclidean(rotation_diff.as_rotvec())
            if translation_diff_sq > max_translation_diff_sq:
                continue
            if angle_sq > max_rotation_diff_sq:
                continue
            result.append(c)

        return result

    @staticmethod
    def _get_line_array(P1, P2, img):
        # https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
        # define local variables for readability

        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(
                    int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[
            (colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[
            itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

        return itbuffer

    # def _get_position(self, depth_map: np.ndarray, gc: GateCandidate):
    #     p = self._parms.depth
    #     h, w = depth_map.shape[:2]
    #     k_h = w // 2 / math.tan(self.__hfov / 2)
    #     k_v = h // 2 / math.tan(self.__vfov / 2)
    #
    #     def get_3d_point(p):
    #         center = np.array([w // 2, h // 2])
    #         pc = np.append(p - center, 1)
    #         x, y = p[0], p[1]
    #         if (not 0 <= x < w) or (not 0 <= y < h):
    #             raise ValueError("Point not in the image")
    #         return np.array([1, pc[0] / k_h, pc[1] / k_v]) * depth_map[y, x]
    #
    #     # get_3d_point = np.vectorize(_get_3d_point)
    #
    #     # depth_map_normalized = (depth_map * 25).astype(np.uint8)
    #     # depth_map_rgb = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_HOT)
    #     def sample_line(l):
    #         v = l[2:] - l[:2]
    #         n = np.linalg.norm(v)
    #         u = v / n * p.sampling_interval
    #         k = np.array([0, 0])
    #         points = []
    #         for i in range(p.start_end_offset,
    #                        int(n / p.sampling_interval) - p.start_end_offset):
    #             points.append(tuple(l[:2] + u * i))
    #             # cv2.circle(depth_map_rgb, points[-1][0], 3, (0,255,0), 1)
    #         return np.array(points).astype(int32)
    #
    #     pts1 = sample_line(gc.l1)
    #     pts2 = sample_line(gc.l2)
    #     ptsc = sample_line(gc.lc)
    #
    #     def map_to_3d(pts2d):
    #         pts3d = np.zeros((len(pts2d), 3))
    #         for i in range(len(pts2d)):  # vectorize
    #             cv2.circle(depth_map, pts2d[i], 5, (0, 0, 255), 1)
    #             x, y = pts2d[i]
    #             if (not 0 <= x < w) or (not 0 <= y < h):
    #                 continue
    #             pts3d[i] = get_3d_point(pts2d[i])
    #         return pts3d
    #
    #     pts3d1 = map_to_3d(pts1)
    #     pts3d2 = map_to_3d(pts2)
    #     pts3dc = map_to_3d(ptsc)
    #
    #     if not len(pts1) or not len(pts2) or not len(ptsc):
    #         return None
    #     pts3d = np.concatenate((pts3d1, pts3d2, pts3dc))
    #     # remove zeros
    #     index = np.where(pts3d[:, 0] == 0.0)
    #     pts3d = np.delete(pts3d, index, axis=0)
    #     pts3d_depth = pts3d[:, 0]
    #     # remove outliers
    #     quantiles = np.quantile(pts3d_depth, [.25, .75])
    #     q1 = quantiles[0]
    #     q3 = quantiles[1]
    #     iqr = q3 - q1
    #     lower = q1 - p.outlier_f * iqr
    #     upper = q3 + p.outlier_f * iqr
    #     index = np.where(
    #         np.logical_or(pts3d_depth < lower, upper < pts3d_depth))
    #     pts3d = np.delete(pts3d, index, axis=0)
    #     # fit plane
    #     try:
    #         plane = Plane.best_fit(pts3d)
    #         # get standard deviation
    #         dev = 0
    #         for point in pts3d:
    #             dev += plane.distance_point(point) ** 2
    #         dev = math.sqrt(dev)
    #         # calculate midpoint (bottom of the center pole) and orientation
    #         mid_line = Line((0, 0, 0), pts3dc[-1])
    #         pos = plane.intersect_line(mid_line)
    #         v1 = gc.l1[:2] - gc.l1[2:]
    #         v2 = gc.l2[:2] - gc.l2[2:]
    #         u1 = v1 / np.linalg.norm(v1)
    #         u2 = v2 / np.linalg.norm(v2)
    #         u = (u1 + u2) / 2
    #         up = plane.project_vector([0, u[0], u[1]])  # points up
    #         up_u = up / np.linalg.norm(up)
    #         z = np.array([0, 0, 1])
    #         [q0, q1, q2] = np.cross(z, up_u)
    #         q3 = np.arccos(np.clip(np.dot(up_u, z), -1.0, 1.0))
    #         return GateDetector.GatePosition(*pos, q0, q1, q2, q3), dev
    #
    #     except ValueError:
    #         return None

    def _find_candidates(self, img):
        """
        Find gate candidates in a color image.
        @param img: OpenCV BGR image
        @return: List of 3D positions of gate candidates.
        """
        p = self._parms

        # Filter in HSV, fill holes and detect contours
        filtered = self._get_color_filter_mask(img,
                                               p.preprocessing.hsv_boundaries)
        floodfilled = self._fill_holes(filtered)
        eroded = self._erode(floodfilled)
        edges = cv2.Canny(eroded, 0, 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Get lines corresponding to gate poles
        lines = self._lines_from_contours(contours)

        self.draw_lines(img, lines, 2, (255, 0, 0))

        if len(lines) == 0:
            return []

        candidates = []
        for i1, i2 in itertools.combinations(range(len(lines)), 2):
            l1 = lines[i1, 0]
            l2 = lines[i2, 0]

            if l1[0] < l2[0] and l1[2] < l2[2]:
                left, right = l1, l2
            elif l2[0] < l1[0] and l2[2] < l1[2]:
                left, right = l2, l1
            else:
                # intersection
                continue

            retval, pos, rot = self._gate_pos_from_side_poles(left, right)
            if retval:
                candidates.append(
                    GateDetector._GateCandidate(left, right, pos, rot))

        return candidates

    def _clip_candidates(self, candidates):
        """
        Eliminates candidates with positions and orientations outside clipping
        range. Considers only GateCandidate.img_pos and GateCandidate.img_rot
        @param candidates: List of _GateCandidate instances
        @return: List of _GateCandidate instances
        """
        p = self._parms

        result = []
        for c in candidates:
            # clip x (depth) and orientation

            # angle of rotation around rotation vector
            angle = np.linalg.norm(c.img_rot.as_rotvec())
            if ((c.img_pos[2] < p.clipping.max_distance) and
                    (math.degrees(angle) < p.clipping.max_angle)):
                result.append(c)

        return result

    def _lines_from_contours(self, contours):
        """
        For an array of CV contours, calculate centerlines of their smallest
        bounding rectangles.
        @param contours: Array of contours
        @return: Array of lines, shape=(n,4), dtype=int32, first point is
        guaranteed to be above the second
        """
        p = self._parms

        lines = []
        for i, c in enumerate(contours):
            (cx, cy), (w, h), th = cv2.minAreaRect(c)

            # Fix the orientation
            if w > h:
                th -= 90.0
            w, h = min(w, h), max(w, h)

            # filter based on minimum size
            if (w == 0 or
                    h < p.contours.min_size or
                    h / w < p.contours.min_hw_ratio):
                continue

            th = -math.radians(th)
            dx = math.sin(th) * h / 2
            dy = math.cos(th) * h / 2

            lines.append([[cx - dx, cy - dy, cx + dx, cy + dy]])

        lines = np.array(lines).astype(int)
        return lines

    def _erode(self, img):
        """
        Erodes the image.
        @param img:
        @return:
        """
        es = self._parms.preprocessing.erosion_ks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (2 * es + 1, 2 * es + 1), (es, es))
        eroded = cv2.erode(img, kernel)
        return eroded

    def _fill_holes(self, img):
        """
        Fills holes in a binary mask.
        @param img (binary mask, 1 channel)
        @return: binary mask with filled holes
        """
        ds = self._parms.preprocessing.dilation_ks

        # dilate the image to close small holes
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                    (2 * ds + 1, 2 * ds + 1),
                                                    (ds, ds))
        dilated = cv2.dilate(img, dilation_kernel)

        # flood fill the image
        im_floodfill = dilated.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        floodfilled = dilated | im_floodfill_inv

        return floodfilled

    def _gate_pos_from_side_poles(self, l1, l2):
        """
        Compute score of how close side poles
        @param l1: Line 1: [x1, y1, x2, y2]
        @param l2: Line 2, should be to the right of line 1
        @return: tuple of retval, translation vec, and rotation
        (as scipy.spatial.transform.Rotation)
        """
        p = self._parms
        image_points = np.array(
            [l1[:2], l2[:2], l2[2:], l1[2:]]
        ).astype(np.float64)
        object_points = np.array(p.geometry.side_poles)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                          self._camera_matrix,
                                          None,
                                          flags=cv2.SOLVEPNP_IPPE)
        # x, y axes same the those of the image (x - right, y - down)
        # z points forward
        if not retval:
            return False, None, None
        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        rot = Rotation.from_rotvec(rvec)
        return True, tvec, rot

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


class GateDetectorNode(GateDetector):
    def __init__(self):
        # print("Initializing Gate Detector...")
        # super().__init__(Parms(rospy.get_param('~gate_detector_parameters')))
        # self.rgb_sub = message_filters.Subscriber('oakd_front/color/image_raw',
        #                                           Image)
        # self.depth_sub = message_filters.Subscriber(
        #     'oakd_front/stereo/depth_map',
        #     Image)
        # self.camera_info_sub = message_filters.Subscriber(
        #     'oakd_front/stereo/left/camera_info', CameraInfo)

        super().__init__(Parms.fromfile(
            '/home/gleb/TAUV-ROS-Packages/src/packages/tauv_config/kingfisher_sim_description/yaml/gate_detector.yaml').gate_detector_parameters)
        self.rgb_sub = message_filters.Subscriber(
            '/kf/vehicle/oakd_front/color/image_raw',
            Image)
        self.depth_sub = message_filters.Subscriber(
            '/kf/vehicle/oakd_front/stereo/depth_map',
            Image)
        self.camera_info_sub = message_filters.Subscriber(
            '/kf/vehicle/oakd_front/stereo/left/camera_info', CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub,
                                                    self.depth_sub,
                                                    self.camera_info_sub], 10)
        self.ts.registerCallback(self.callback)
        self.detection_pub = rospy.Publisher('gate_detections', PoseArray,
                                             queue_size=10)
        self.bridge = cv_bridge.CvBridge()
        # print("Done.")

    def callback(self, rgb: Image, depth: Image, camera_info: CameraInfo):
        # Update FOV
        self.set_camera_matrix(camera_info.K)

        # Run detection
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_depth = self.bridge.imgmsg_to_cv2(depth,
                                             desired_encoding="passthrough")
        detections = self.detect(cv_rgb, cv_depth)
        # cv2.imshow('rgb', cv_rgb)
        # cv2.waitKey(1)
        pose_array = PoseArray()
        pose_array.header.stamp = rgb.header.stamp
        pose_array.header.frame_id = "kf/vehicle"
        for pos in detections:
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
