import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "camera_cal"
OUTPUT_DIR = "output_images/"
# Verify the existing directory and saving the images
# Calibration of images

# Variables
################################################################################
nx = 9
ny = 6
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.
img_err = []  # Images which were failed to open

################################################################################
# Function to clean the output image directory
def clean_output_dir():
    for file in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, file)
        # print(file_path)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
print(clean_output_dir())


################################################################################
def calibrate_camera(input_dir, output_dir, nx, ny):
    # Clean the output directory
    for id, name in enumerate(os.listdir(input_dir)):
        img = cv2.imread(input_dir + "/" + name)
        # print(id, name)
        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cornerSubPix use the criteria function to fine tune the images
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # Verify if the corners were returned
        if ret == True:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            f.tight_layout()
            ax1.imshow(cv2.cvtColor(cv2.imread(INPUT_DIR + "/" + name), cv2.COLOR_BGR2RGB))
            ax1.set_title("Original:: " + name, fontsize=18)
            ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax2.set_title("Corners:: " + name, fontsize=18)
            f.savefig(output_dir + "/output_" + str(time.time()) + ".jpg")
        else:
            img_err.append(name)

print(calibrate_camera(INPUT_DIR, OUTPUT_DIR, nx, ny))

#########################################################################################
# Distortion correction
def undistort(img_name, objpoints, imgpoints):
    img = cv2.imread(img_name)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    return undist
#########################################################################################
