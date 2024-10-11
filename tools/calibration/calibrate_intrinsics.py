import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog='Calibrate single camera intrinsics',
    description='Takes a folder of charuco images',
    epilog=''
)

parser.add_argument('image_path', type=Path)
parser.add_argument('charuco_width', type=int)
parser.add_argument('charuco_height', type=int)
parser.add_argument('square_size_mm', type=float)
parser.add_argument('marker_size_mm', type=float)

if __name__ == '__main__':
    args = parser.parse_args()

    board = cv2.aruco.CharucoBoard((args.charuco_width, args.charuco_height),
                                   args.square_size_mm,
                                   args.marker_size_mm,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    board.setLegacyPattern(True)

    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)

    images = args.image_path.glob('*.png')

    all_object_points = []
    all_image_points = []

    n = 0
    for image_path in tqdm(images, desc="Processing Images: "):
        image = cv2.imread(image_path)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)

        if charuco_ids is None:
            charuco_ids = []
        if charuco_corners is None:
            charuco_corners = []

        if len(charuco_corners) and len(charuco_ids):
            obj_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
            all_object_points.append(obj_points)
            all_image_points.append(image_points)
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
        cv2.imshow('image', image)
        cv2.waitKey(1)


    image_size = image.shape[:2]
    camera_mat = np.zeros((3,3))
    dist_coeffs = np.zeros((14,))

    print('Calibrating intrinsics...')
    (rep_error,
     camera_mat,
     dist_coeffs, _, _) = cv2.calibrateCamera(all_object_points, all_image_points, image_size,
                                              camera_mat, dist_coeffs)

    print(f'Calibration complete, reprojection error = {rep_error}')
    print('\n\n\n INTRINSICS MATRIX:')
    print(camera_mat)
    print('\n\n\n DISTORTION COEFFS:')
    print(dist_coeffs)

