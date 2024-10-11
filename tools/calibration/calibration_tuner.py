import cv2
import numpy as np
import sys


# Define a function to update the displayed image
def update_image(image, detector, board):
    # Detect markers
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)

    # If markers are detected, estimate the pose
    if charuco_ids is not None:
        obj_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
        cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

    # Display the number of detected markers
    num_markers = charuco_ids.shape[0] if charuco_ids is not None else 0
    cv2.putText(image, f'Detected Markers: {num_markers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image


# Callback function for the sliders (optional)
def nothing(x):
    pass


# Main function to run the application
def main(image_path):
    # Set up the ChArUco board and dictionary
    board = cv2.aruco.CharucoBoard((41, 23),
                                   4.521,
                                   3.41,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    board.setLegacyPattern(True)

    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)

    # Create a window
    cv2.namedWindow('ChArUco Detector')

    # Create sliders for parameters (optional; adjust as needed)
    cv2.createTrackbar('adaptiveThreshConstant', 'ChArUco Detector', 7, 30, nothing)
    cv2.createTrackbar('adaptiveThreshWinSizeMax', 'ChArUco Detector', 23, 100, nothing)
    cv2.createTrackbar('adaptiveThreshWinSizeMin', 'ChArUco Detector', 3, 20, nothing)
    cv2.createTrackbar('aprilTagCriticalRad', 'ChArUco Detector', 10, 180, nothing)
    cv2.createTrackbar('aprilTagMaxLineFitMse', 'ChArUco Detector', 3, 100, nothing)
    cv2.createTrackbar('aprilTagMaxNmaxima', 'ChArUco Detector', 10, 100, nothing)
    cv2.createTrackbar('aprilTagMinClusterPixels', 'ChArUco Detector', 5, 50, nothing)
    cv2.createTrackbar('aprilTagMinWhiteBlackDiff', 'ChArUco Detector', 5, 255, nothing)

    # Load the image
    image = cv2.imread(image_path)

    while True:
        # Get current parameter values from all sliders
        detector_params.adaptiveThreshConstant = cv2.getTrackbarPos('adaptiveThreshConstant', 'ChArUco Detector')
        detector_params.adaptiveThreshWinSizeMax = cv2.getTrackbarPos('adaptiveThreshWinSizeMax', 'ChArUco Detector')
        detector_params.adaptiveThreshWinSizeMin = cv2.getTrackbarPos('adaptiveThreshWinSizeMin', 'ChArUco Detector')
        detector_params.aprilTagCriticalRad = cv2.getTrackbarPos('aprilTagCriticalRad', 'ChArUco Detector')
        detector_params.aprilTagMaxLineFitMse = cv2.getTrackbarPos('aprilTagMaxLineFitMse', 'ChArUco Detector')
        detector_params.aprilTagMaxNmaxima = cv2.getTrackbarPos('aprilTagMaxNmaxima', 'ChArUco Detector')
        detector_params.aprilTagMinClusterPixels = cv2.getTrackbarPos('aprilTagMinClusterPixels', 'ChArUco Detector')
        detector_params.aprilTagMinWhiteBlackDiff = cv2.getTrackbarPos('aprilTagMinWhiteBlackDiff', 'ChArUco Detector')

        # Update the image with detected markers
        detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
        updated_image = update_image(image.copy(), detector, board)

        # Show the image
        cv2.imshow('ChArUco Detector', updated_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python charuco_detector.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
