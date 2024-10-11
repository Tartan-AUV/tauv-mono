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
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    board.setLegacyPattern(True)

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.minMarkerPerimeterRate = 0.01
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.adaptiveThreshConstant = 3
    detector_params.adaptiveThreshWinSizeMin = 10
    detector_params.adaptiveThreshWinSizeMax = 30
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



def analyze_charuco(self, images, scale_req=False, req_resolution=(800, 1280)):
    """
    Charuco base pose estimation.
    """
    # print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    all_marker_corners = []
    all_marker_ids = []
    all_recovered = []
    # decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
    count = 0
    skip_vis = False

    board = cv2.aruco.CharucoBoard((args.charuco_width, args.charuco_height),
                                   args.square_size_mm,
                                   args.marker_size_mm,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    board.setLegacyPattern(True)
    for im in images:
        if self.traceLevel == 3 or self.traceLevel == 10:
            print("=> Processing image {0}".format(im))
        img_pth = Path(im)
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        expected_height = gray.shape[0]*(req_resolution[1]/gray.shape[1])

        if scale_req and not (gray.shape[0] == req_resolution[0] and gray.shape[1] == req_resolution[1]):
            if int(expected_height) == req_resolution[0]:
                # resizing to have both stereo and rgb to have same
                # resolution to capture extrinsics of the rgb-right camera
                gray = cv2.resize(gray, req_resolution[::-1],
                                  interpolation=cv2.INTER_CUBIC)
            else:
                # resizing and cropping to have both stereo and rgb to have same resolution
                # to calculate extrinsics of the rgb-right camera
                scale_width = req_resolution[1]/gray.shape[1]
                dest_res = (
                    int(gray.shape[1] * scale_width), int(gray.shape[0] * scale_width))
                gray = cv2.resize(
                    gray, dest_res, interpolation=cv2.INTER_CUBIC)
                if gray.shape[0] < req_resolution[0]:
                    raise RuntimeError("resizeed height of rgb is smaller than required. {0} < {1}".format(
                        gray.shape[0], req_resolution[0]))
                # print(gray.shape[0] - req_resolution[0])
                del_height = (gray.shape[0] - req_resolution[0]) // 2
                # gray = gray[: req_resolution[0], :]
                gray = gray[del_height: del_height + req_resolution[0], :]

            count += 1
        marker_corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, self.aruco_dictionary)
        marker_corners, ids, refusd, recoverd = cv2.aruco.refineDetectedMarkers(gray, self.board,
                                                                                marker_corners, ids, rejectedCorners=rejectedImgPoints)
        if self.traceLevel == 2 or self.traceLevel == 4 or self.traceLevel == 10:
            print('{0} number of Markers corners detected in the image {1}'.format(
                len(marker_corners), img_pth.name))
        if len(marker_corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, ids, gray, self.board, minMarkers = 1)

            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:

                cv2.cornerSubPix(gray, charuco_corners,
                                 winSize=(5, 5),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
                allCorners.append(charuco_corners)  # Charco chess corners
                allIds.append(charuco_ids)  # charuco chess corner id's
                all_marker_corners.append(marker_corners)
                all_marker_ids.append(ids)
            else:
                print(im)
                raise RuntimeError("Failed to detect markers in the image")
        else:
            print(im + " Not found")
            raise RuntimeError("Failed to detect markers in the image")
        if self.traceLevel == 2 or self.traceLevel == 4 or self.traceLevel == 10:
            rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(rgb_img, marker_corners, ids, (0, 0, 255))
            cv2.aruco.drawDetectedCornersCharuco(rgb_img, charuco_corners, charuco_ids, (0, 255, 0))

            if rgb_img.shape[1] > 1920:
                rgb_img = cv2.resize(rgb_img, (0, 0), fx=0.7, fy=0.7)
            if not skip_vis:
                name = img_pth.name + ' - ' + "marker frame"
                cv2.imshow(name, rgb_img)
                k = cv2.waitKey(0)
                if k == 27: # Esc key to skip vis
                    skip_vis = True
            cv2.destroyAllWindows()
    # imsize = gray.shape[::-1]
    return allCorners, allIds, all_marker_corners, all_marker_ids, gray.shape[::-1], all_recovered
