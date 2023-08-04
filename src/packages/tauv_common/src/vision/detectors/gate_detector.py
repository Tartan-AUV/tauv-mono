#!/usr/bin/env python3

# Author: Gleb Ryabtsev, 2023

import cv2
import numpy as np
from skspatial.objects import Plane, Line
from scipy.spatial.transform import Rotation

import math
import itertools

from collections import namedtuple
from dataclasses import dataclass
from tauv_util.parms import Parms

from typing import List

class GateDetector:
    GatePosition = namedtuple('GatePosition', 'x y z q0 q1 q2 q3')

    @dataclass
    class _GateCandidate:
        left_pole_img: np.ndarray
        right_pole_img: np.ndarray
        img_translation: np.ndarray
        img_rotation: Rotation
        depth_translation: np.ndarray = None
        depth_rotation: Rotation = None
        depth_plane_dev: float = None

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
        @return: None
        """
        self._camera_matrix = np.array(camera_matrix).reshape(3, 3)
        # get inverse
        self._camera_matrix_inv = self._inv_camera_matrix(self._camera_matrix)

    def detect(self, color: np.array, depth: np.array):
        """
        Run the gate detector on the color and depth data and return the list
        of possible gate positions
        @param color: BGR color image
        @param depth: Single channel floating point depth image
        @return:
        """

        print('detecting')
        p = self._parms

        # find gate candidates on the rgb image
        candidates = self._find_candidates(color)

        # clip position and orientation
        candidates = self._clip_candidates(candidates)

        self._calculate_depth_poses(candidates, depth)

        candidates = self._filter_by_depth_dev(candidates)

        candidates = self._filter_by_pose_diff(candidates)

        detections = []
        for c in candidates:
            d = self.GatePosition(*c.depth_translation, *c.depth_rotation.as_quat())
            detections.append(d)
        return detections

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

        cv2.imshow("filtered", filtered)
        print(p.preprocessing.hsv_boundaries)

        floodfilled = self._fill_holes(filtered)
        eroded = self._erode(floodfilled)
        edges = cv2.Canny(eroded, 0, 255)



        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Get lines corresponding to gate poles
        lines = self._lines_from_contours(contours)

        self._draw_lines(img, lines, 2, (255, 0, 0))

        cv2.imshow('lines', img)
        cv2.waitKey(1)

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

            retval, tvec, rot = self._pose_from_side_poles(left, right)
            if retval:
                candidates.append(
                    GateDetector._GateCandidate(left, right, tvec, rot))

        return candidates

    def _filter_by_depth_dev(self,
                             candidates: List[_GateCandidate]):
        """
        Filters a list of _GateCandidates based on the deviation from the
        best fit plane
        @param candidates: List of GateCandidates
        @return: List of GateCandidates
        """
        p = self._parms

        result = []
        for c in candidates:
            if c.depth_plane_dev is None:
                # Could not fit plane, so assuming candidate is invalid
                continue
            if c.depth_plane_dev < p.depth.dev_cutoff:
                result.append(c)

        return result

    def _calculate_depth_poses(self,
                               candidates: List[_GateCandidate],
                               depth_map: np.ndarray):
        """
        Calculate the depth-based translations and rotations of _GateCandidate
        instances
        @param candidates: Array of gate candidates
        @param depth_map: Depth image (1 channel)
        @return: None
        """
        p = self._parms.depth

        for gc in candidates:

            # Sample lines from the depth map
            left_top, left_bottom = gc.left_pole_img[:2], gc.left_pole_img[2:]
            right_top, right_bottom = gc.right_pole_img[:2], gc.right_pole_img[2:]
            left_2d = self._get_line_array(left_top, left_bottom, depth_map)
            right_2d = self._get_line_array(right_top, right_bottom, depth_map)
            pts_2d = np.concatenate((left_2d, right_2d), axis=0)
            pts_2d = pts_2d[::p.sampling_interval]

            # Remove points with depth=0
            depth = pts_2d[:, 2]
            index = np.where(depth == 0.0)
            pts_2d = np.delete(pts_2d, index, axis=0)

            # Remove outliers
            depth = pts_2d[:, 2]
            if len(depth) < 3:
                continue
            quantiles = np.quantile(depth, [.25, .75])
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower = q1 - p.outlier_f * iqr
            upper = q3 + p.outlier_f * iqr
            index = np.where(np.logical_or(depth < lower, upper < depth))
            pts_2d = np.delete(pts_2d, index, axis=0)

            # Prepare array of 2D points for matrix multiplication
            # pts_2d_4: [[u*depth, v*depth, depth, 1], ...]
            pts_2d_4 = np.zeros((len(pts_2d), 4), dtype=np.float64)
            pts_2d_4[:, 0] = pts_2d[:, 0] * pts_2d[:, 2]
            pts_2d_4[:, 1] = pts_2d[:, 1] * pts_2d[:, 2]
            pts_2d_4[:, 2] = pts_2d[:, 2]
            pts_2d_4[:, 3] = 1

            # get 3D points (camera_matrix_inv x pts_2d_4)
            mat = self._camera_matrix_inv
            pts_3d = np.matmul(mat, pts_2d_4.transpose())
            pts_3d = pts_3d.transpose()

            # fit a plane
            try:
                plane = Plane.best_fit(pts_3d)
            except ValueError:
                continue

            # calculate deviation
            a, b, c, d = plane.cartesian()
            normal = np.array([a, b, c])
            normal_length = np.linalg.norm(normal)
            normal = normal / normal_length
            d /= normal_length
            plane_pt = normal * d * -1.0
            squared_distances = np.abs((pts_3d - plane_pt).dot(normal))
            gc.depth_plane_dev = np.sum(squared_distances) / len(squared_distances)

            # calculate gate position based on the plane
            # get the gate center point vector
            center_2d = (left_top + left_bottom + right_top + right_bottom) / 4.0
            center_2d_4 = np.concatenate((center_2d, [1, 1]))
            center_3d_vec = np.matmul(mat, center_2d_4)[:3]
            center_3d_line = Line([0., 0., 0.], center_3d_vec)
            gc.depth_translation = np.array(plane.intersect_line(center_3d_line))

            # calculate rotation
            if normal[2] < 0.0:
                normal *= (-1.0)
            z_unit = [0, 0, 1]
            x, y, z = np.cross(z_unit, normal)
            w = 1.0 + np.dot(normal, z_unit)
            gc.depth_rotation = Rotation.from_quat((x, y, z, w))

    def _filter_by_pose_diff(self, candidates):
        """
        Filters candidates list by the difference between poses calcualted from
        the depth and the color images
        @param candidates: List of _GateCandidate instances
        @return: List of _GateCandidate instances
        """
        p = self._parms

        # Prepare thresholds (faster to compare squares)
        max_translation_diff_sq = p.depth.max_translation_diff ** 2
        max_rotation_diff_sq = math.radians(p.depth.max_rotation_diff) ** 2

        # Squared norm function
        sq_euclidean = lambda x: np.inner(x, x)

        result = []
        for c in candidates:
            if c.depth_translation is None or c.depth_rotation is None:
                continue
            translation_diff_sq = sq_euclidean(c.img_translation - c.depth_translation)
            rotation_diff = c.depth_rotation.inv() * c.img_rotation
            angle_sq = sq_euclidean(rotation_diff.as_rotvec())
            if translation_diff_sq > max_translation_diff_sq:
                continue
            if angle_sq > max_rotation_diff_sq:
                continue
            result.append(c)

        return result

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
            angle = np.linalg.norm(c.img_rotation.as_rotvec())
            if ((c.img_translation[2] < p.clipping.max_distance) and
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
        @param img: Binary image
        @return: Binary image, eroded
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

    def _pose_from_side_poles(self, l1, l2):
        """
        Compute the gate pose from two lines in the image corresponding to the
        side poles.
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
    def _inv_camera_matrix(mat):
        """
        Calculate the inverse of a camera matrix
        @param mat: 3x3 np camera matrix
        @return: 3x4 inverted camera matrix
        """
        inverse_3x3 = np.linalg.inv(mat)
        inverse_3x4 = np.concatenate((inverse_3x3, [[0], [0], [0]]), axis=1)
        return inverse_3x4

    @staticmethod
    def _get_color_filter_mask(img, hsv_boundaries):
        """
        Set all pixels with hsv values outside the given range to zero.
        @param: img: Image in BGR
        @param: hsv_boundaries [[[hmin, smin, vmin], [hmax, smax, vmax]], ...]
        @return: Binary mask of image areas within color ranges
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

    @staticmethod
    def _draw_lines(img, lines, width, color):
        if lines is None:
            return
        for line in lines:
            line = line[0]
            if len(line) == 4:
                cv2.line(img, tuple(line[0:2]), tuple(line[2:4]), color, width)
            else:
                raise NotImplementedError()
