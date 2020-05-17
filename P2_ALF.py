import glob
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

########################################################################################################################
INPUT_DIR = "camera_cal/"
OUTPUT_DIR = "output_images/"
DIR_TEST_IMG = "test_images/"
TEST_IMAGES = glob.glob(DIR_TEST_IMG + "test*.jpg")


# Verify the existing directory and saving the images
# Calibration of images
########################################################################################################################
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


clean_output_dir()
print("Output Image Directory Cleaned Successfully ! ... ")
########################################################################################################################
# Define List and Variables for corners and points
nx = 9
ny = 6
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
objpoints = []  # 3d points in real world space (x,y,z) co-ordinates
imgpoints = []  # 2d points in image plane.
img_err = []  # Images which were failed to open will be stored here


########################################################################################################################
def calibrate_camera(input_dir, output_dir, nx, ny):
    # Clean the output directory
    for id, name in enumerate(os.listdir(input_dir)):
        img = cv2.imread(input_dir + name)
        # print(id, name)
        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cornerSubPix use the criteria function to fine tune the images
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
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
            f.savefig(output_dir + "output_calibrate_camera_" + str(time.time()) + ".jpg")
        else:  # saving the failed to open images in the list
            img_err.append(name)


calibrate_camera(INPUT_DIR, OUTPUT_DIR, nx, ny)
print("Camera Calibration Completed! ... ")


########################################################################################################################
# Distortion correction passing img,object and image points and returning the undisorted images points
def undistort(img_name, objpoints, imgpoints):
    img = cv2.imread(img_name)
    # print("UND_TEST", img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


########################################################################################################################
# Undisort Test Single_Images and save to directory
undist = undistort(INPUT_DIR + "calibration10.jpg", objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(cv2.imread(INPUT_DIR + "/calibration10.jpg"), cv2.COLOR_BGR2RGB))
ax1.set_title("Original_Image:: calibration10.jpg", fontsize=18)
ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
ax2.set_title("Undistorted_Image:: calibration10.jpg", fontsize=18)
f.savefig(OUTPUT_DIR + "output_undistort_single_test_file_" + str(time.time()) + ".jpg")

########################################################################################################################
# Undisort all the images in the test folder
########################################################################################################################

def undisort_images(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


for image in TEST_IMAGES:
    undist = undistort(image, objpoints, imgpoints)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    ax2.set_title("Undistorted:: " + image, fontsize=18)
    f.savefig(OUTPUT_DIR + "output_undistort_" + str((time.time())) + ".jpg")


########################################################################################################################


# Prespective Transformation
def transform_image(img, offset=250, src=None, dst=None, lane_width=9):
    image_dimension = (img.shape[1], img.shape[0])
    # Copy the Image
    out_img_orig = np.copy(img)
    # Define the area
    leftupper = (585, 460)
    rightupper = (705, 460)
    leftlower = (210, img.shape[0])
    rightlower = (1080, img.shape[0])

    warped_leftupper = (offset, 0)
    warped_rightupper = (offset, img.shape[0])
    warped_leftlower = (img.shape[1] - offset, 0)
    warped_rightlower = (img.shape[1] - offset, img.shape[0])
    # define the color to be drawn on the border of the image
    color_red = [0, 0, 255]
    color_cyan = [255, 255, 0]
    lane_width = 9

    if src is not None:
        src = src
    src = np.float32([leftupper, leftlower, rightupper, rightlower])

    if dst is not None:
        dst = dst
    dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])

    cv2.line(out_img_orig, leftlower, leftupper, color_red, lane_width)
    cv2.line(out_img_orig, leftlower, rightlower, color_red, lane_width * 2)
    cv2.line(out_img_orig, rightupper, rightlower, color_red, lane_width)
    cv2.line(out_img_orig, rightupper, leftupper, color_cyan, lane_width)

    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image
    warped = cv2.warpPerspective(img, M, image_dimension, flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    # (warped)
    out_warped_img = np.copy(warped)

    cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_red, lane_width)
    cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_red, lane_width * 2)
    cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_red, lane_width)
    cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_cyan, lane_width)

    return warped, M, minv, out_img_orig, out_warped_img


########################################################################################################################
# Run the function
for image in TEST_IMAGES:
    img = cv2.imread(image)
    warped, M, minv, out_img_orig, out_warped_img = transform_image(img)  # Calling the transform Image Output
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(out_img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(cv2.cvtColor(out_warped_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped:: " + image, fontsize=18)
    f.savefig(OUTPUT_DIR + "output_transform_image_" + str(time.time()) + ".jpg")


########################################################################################################################
# Gradient and Color Transformation
# Applying Sobel Operator and returning binary image in the return
def abs_sobel_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), return_grad=False,
                     direction='x'):  # Default Direction is X
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Applying Sobel operator in X Direction
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    # Sobel in Y direction
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in y

    if return_grad == True:
        return grad

    abs_sobel = np.absolute(
        grad)  # Absolute x/y based on the input in the function derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Applying the threshold in the image and returning the output
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1

    return grad_binary


########################################################################################################################
# Running the function on the first Undisorted Image and getting the binary Image in the X Direction
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = abs_sobel_thresh(img, sobel_kernel=3, mag_thresh=(30, 100), direction='x')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
# plt.show()
ax2.set_title("Transformed:: " + image, fontsize=18)
# plt.show()

#  Running the function on the first Undisorted Image and getting the binary Image in the X Direction
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = abs_sobel_thresh(img, sobel_kernel=3, mag_thresh=(30, 120), direction='y')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)


############################################################################################################################
# Calulating the magnitude of the gradient threshold
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    xgrad = abs_sobel_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True)  # X Direction
    ygrad = abs_sobel_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True,
                             direction='y')  # In Y direction

    magnitude = np.sqrt(np.square(xgrad) + np.square(ygrad))  # Calculating the magnitude of Gradient
    abs_magnitude = np.absolute(magnitude)
    scaled_magnitude = np.uint8(255 * abs_magnitude / np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude < mag_thresh[1])] = 1

    return mag_binary


# Run the function
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = mag_threshold(img, mag_thresh=(30, 100))  # Get Binary Image
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary,
                                                                                   offset=300)  # Transform the Image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)


########################################################################################################################
# Calculating the Direction of the Gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale Images conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculating the absolute value  X and Y gradients
    # xgrad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # ygrad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    xabs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    yabs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    grad_dir = np.arctan2(yabs, xabs)
    # Applying the threshold and creating the binary image
    binary_output = np.zeros_like(grad_dir).astype(np.uint8)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir < thresh[1])] = 1
    return binary_output


########################################################################################################################
# Calulating the RGB Threshold and returning the binary Image
def calulate_RGB_threshold_image(img, channel='R', thresh=(0, 255)):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel == 'R':
        bin_img = img1[:, :, 0]
    if channel == 'G':
        bin_img = img1[:, :, 1]
    if channel == 'B':
        bin_img = img1[:, :, 2]

    binary_img = np.zeros_like(bin_img).astype(np.uint8)
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1

    return binary_img


########################################################################################################################
# Calculate the RGB Threshold on the First Binary Image for R Channel
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = calulate_RGB_threshold_image(img, thresh=(230, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)
# ***************************************************************************************************************
# Calculate the RGB Threshold on the First Binary Image for G Channel
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = calulate_RGB_threshold_image(img, thresh=(200, 255), channel='G')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)
# ***************************************************************************************************************
# Calculate the RGB Threshold on the First Binary Image for B Channel
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = calulate_RGB_threshold_image(img, thresh=(185, 255), channel='B')
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)


########################################################################################################################
##################### HLS Color Threshold #################################################
# Calulate the threshold the h-channel of HLS
def get_hls_hthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_channel = hls_img[:, :, 0]
    # Matching the condition for the criteria
    binary_output = np.zeros_like(h_channel).astype(np.uint8)
    binary_output[(h_channel >= thresh[0]) & (h_channel < thresh[1])] = 1
    # return the image output
    return binary_output


########################################################################################################################
# Calulate the threshold the l-channel of HLS
def get_hls_lthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls_img[:, :, 1]
    # Matching the condition for the criteria
    binary_output = np.zeros_like(l_channel).astype(np.uint8)
    binary_output[(l_channel >= thresh[0]) & (l_channel < thresh[1])] = 1
    # return the image output
    return binary_output


########################################################################################################################
# Calculate the threshold the s-channel of HLS
def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls_img[:, :, 2]
    # Matching the condition for the criteria
    binary_output = np.zeros_like(s_channel).astype(np.uint8)
    binary_output[(s_channel >= thresh[0]) & (s_channel < thresh[1])] = 1
    # return the image output
    return binary_output


# Run the function on the Test Image
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
# combined_binary = get_hls_hthresh_img(img, thresh=(201, 255))
combined_binary = get_hls_lthresh_img(img, thresh=(201, 255))
# combined_binary = get_hls_sthresh_img(img, thresh=(201, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)

########################################################################################################################

# Run the function
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = get_hls_sthresh_img(img, thresh=(150, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)


########################################################################################################################
# The Lab color space is quite different from the RGB color space. In RGB color
# space the color information is separated into three channels but the same
# three channels also encode brightness information
# On the other hand, in Lab color space, the L channel is independent of color information and encodes brightness only.
# The other two channels encode color.
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
########################################################################################################################
# calculate b channel in LAB color Space
def get_lab_bthresh_img(img, thresh=(0, 255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab_img[:, :, 2]
    # Matching the condition for the criteria
    bin_op = np.zeros_like(b_channel).astype(np.uint8)
    bin_op[(b_channel >= thresh[0]) & (b_channel < thresh[1])] = 1
    return bin_op


# Test the Image
img = undistort(TEST_IMAGES[0], objpoints, imgpoints)
combined_binary = get_lab_bthresh_img(img, thresh=(147, 255))
warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original:: " + image, fontsize=18)
ax2.imshow(warped, cmap='gray')
ax2.set_title("Transformed:: " + image, fontsize=18)


########################################################################################################################
def get_binary_image(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0, 255), r_thresh=(0, 255),
                     s_thresh=(0, 255), b_thresh=(0, 255), g_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    r_binary = calulate_RGB_threshold_image(img, thresh=r_thresh)
    # Threshold G color channel
    g_binary = calulate_RGB_threshold_image(img, thresh=g_thresh, channel='G')
    # Threshold B in LAB
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)
    # Threshold color channel
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (b_binary == 1) | (g_binary == 1)] = 1
    # combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1) | (g_binary == 1)] = 1
    # Return the output Binary Image which matches the above criteria
    return combined_binary


########################################################################################################################
# Testing the threshing
# Defining threshold values for getting the binary images
kernel_size = 5  # Kernel Size
mag_thresh = (30, 100)  # Magnitude of the threshold
r_thresh = (230, 255)  # Threshold value of R in RGB Color space
s_thresh = (165, 255)  # Threshold value of S in HLS Color space
b_thresh = (160, 255)  # Threshold value of B in LAB Color space
g_thresh = (210, 255)  # Threshold value of G in RGB Color space

for image_name in TEST_IMAGES:
    img = undistort(image_name, objpoints, imgpoints)
    combined_binary = get_binary_image(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh,
                                       s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(combined_binary, cmap='gray')
    ax2.set_title("Threshold Binary:: " + image, fontsize=18)
    # Saving the output in the output directory
    f.savefig(OUTPUT_DIR + "output_get_bin_images_" + str(time.time()) + ".jpg")

########################################################################################################################
for image in TEST_IMAGES:
    img = cv2.imread(image)
    warped, M, minv, out_img_orig, out_warped_img = transform_image(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(out_img_orig, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(cv2.cvtColor(out_warped_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Warped:: " + image, fontsize=18)
    f.savefig(OUTPUT_DIR + "output_transform_image_" + str(time.time()) + ".jpg")

for image in TEST_IMAGES:
    img = undistort(image, objpoints, imgpoints)
    combined_binary = get_binary_image(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                       r_thresh=r_thresh, s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title("Transformed:: " + image, fontsize=18)
    f.savefig(OUTPUT_DIR + "output_get_binary_image_" + str(time.time()) + ".jpg")


########################################################################################################################
########################### Lane  Line Detection and Ploynomial fitting ################################################

def find_lines(warped_img, nwindows=9, margin=80, minpix=40):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height

        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img


########################################################################################################################
# Fitting a polynomial for a window size, margin, ad minimum number of pixel in the window area
def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50, show=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
        = find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # defining a value if left and right are incorrect and values are None
        print("Unable to plot printing the default values")
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]  # Filling RED Color on the lane region
    out_img[righty, rightx] = [0, 0, 255]  # Filling the BLUE Color in the Lane Region

    # Plots the left and right polynomials on the lane lines
    if show == True:
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    # Returning the values required for the  polynimal fir
    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img


########################################################################################################################
# Searching  the values around the polynmonial for the
def search_around_poly(binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
    # Getting the activated pixel
    nonzero = binary_warped.nonzero()  # Non Zero tuples indices for X and y
    nonzeroy = np.array(nonzero[0])  # Non Zero Indices for X
    nonzerox = np.array(nonzero[1])  # Non Zero Indices for Y

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Generating the lef and right lane pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fitting  a 2nd  order polynomial for each pixel
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit second order polynomial to for for points on real world
    left_lane_indices = np.polyfit(lefty * ymtr_per_pixel, leftx * xmtr_per_pixel, 2)
    right_lane_indices = np.polyfit(righty * ymtr_per_pixel, rightx * xmtr_per_pixel, 2)

    return left_fit, right_fit, left_lane_indices, right_lane_indices


# -----------------------------------------------------------------------------------------------------------------------

left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped,
                                                                                                            nwindows=11)
plt.imshow(out_img)
plt.savefig(OUTPUT_DIR + "output_fit_polynomial_" + str(time.time()) + ".jpg")


########################################################################################################################
# Measuring  Radius of the curvature
def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    # Generating the data to represent lane-line pixels
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty * ymtr_per_pixel, left_fitx * xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty * ymtr_per_pixel, right_fitx * xmtr_per_pixel, 2)

    # getting left and right curvature
    left_curverad = ((1 + (
            2 * left_fit_cr[0] * y_eval * ymtr_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    # print(left_curverad)
    right_curverad = ((1 + (
            2 * right_fit_cr[0] * y_eval * ymtr_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # print(right_curverad)
    # returning the left and right lane radius
    return (left_curverad, right_curverad)


########################################################################################################################
# Finding distance from Center
def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ymax = img.shape[0] * ymtr_per_pixel
    center = img.shape[1] / 2

    lineLeft = left_fit[0] * ymax ** 2 + left_fit[1] * ymax + left_fit[2]
    lineRight = right_fit[0] * ymax ** 2 + right_fit[1] * ymax + right_fit[2]

    # Finding the Mid Value and distance from the center
    mid = lineLeft + (lineRight - lineLeft) / 2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0.:
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))

    return message


########################################################################################################################
# Drawing lines in the region
def draw_lines(img, left_fit, right_fit, minv):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Find left and right points.
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the Green value lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]),
                                     flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)


########################################################################################################################
# Displaying the curvature in the output image
def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt = dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)

    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]), 0)
    cv2.putText(out_img, 'Average Lane Curvature: {:.2f} m'.format(avg_rad),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return out_img


for image in TEST_IMAGES:
    img = undistort(image, objpoints, imgpoints)

    combined_binary = get_binary_image(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh,
                                       s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh)
    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)

    xmtr_per_pixel = 3.7 / 800
    ymtr_per_pixel = 30 / 720

    left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped,
                                                                                                                nwindows=12,
                                                                                                                show=False)
    lane_img = draw_lines(img, left_fit, right_fit, minv)
    out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original:: " + image, fontsize=18)
    ax2.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Lane:: " + image, fontsize=18)
    f.savefig(OUTPUT_DIR + "output_curvature_" + str(time.time()) + ".jpg")


########################################################################################################################################
# Defining a class and initializing and storing the values

class Video_Pipeline():
    def __init__(self, max_counter):
        self.current_fit_left = None
        self.best_fit_left = None
        self.history_left = [np.array([False])]
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right = [np.array([False])]
        self.counter = 0
        self.max_counter = 1
        self.src = None
        self.dst = None

    def set_presp_indices(self, src, dst):
        self.src = src
        self.dst = dst

    def reset(self):
        self.current_fit_left = None
        self.best_fit_left = None
        self.history_left = [np.array([False])]
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right = [np.array([False])]
        self.counter = 0

    def update_fit(self, left_fit, right_fit):
        if self.counter > self.max_counter:
            self.reset()
        else:
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            self.history_left.append(left_fit)
            self.history_right.append(right_fit)
            self.history_left = self.history_left[-self.max_counter:] if len(
                self.history_left) > self.max_counter else self.history_left
            self.history_right = self.history_right[-self.max_counter:] if len(
                self.history_right) > self.max_counter else self.history_right
            self.best_fit_left = np.mean(self.history_left, axis=0)
            self.best_fit_right = np.mean(self.history_right, axis=0)

    def process_image(self, image):
        img = undisort_images(image, objpoints, imgpoints)

        combined_binary = get_binary_image(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                           r_thresh=r_thresh, s_thresh=s_thresh, b_thresh=b_thresh, g_thresh=g_thresh)

        if self.src is not None or self.dst is not None:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary,
                                                                                               src=self.src,
                                                                                               dst=self.dst)
        else:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)

        xmtr_per_pixel = 3.7 / 800
        ymtr_per_pixel = 30 / 720

        if self.best_fit_left is None and self.best_fit_right is None:
            left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(
                warped, nwindows=15, show=False)
        else:
            left_fit, right_fit, left_lane_indices, right_lane_indices = search_around_poly(warped, self.best_fit_left,
                                                                                            self.best_fit_right,
                                                                                            xmtr_per_pixel,
                                                                                            ymtr_per_pixel)

        self.counter += 1

        lane_img = draw_lines(img, left_fit, right_fit, unwarp_matrix)
        out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

        self.update_fit(left_fit, right_fit)
        return out_img


########################################################################################################################
######################### INPUT Files for the Video ################################
PRJCT_VIDEO = "project_video.mp4"
CHLNG_VIDEO = "challenge_video.mp4"
HARDC_VIDEO = "harder_challenge_video.mp4"
VIDEO_OPDIR = "output_video/"


########################################################################################################################
########################## Function to create Video output ###################################
def create_video_output(input_video_file, output_dir=VIDEO_OPDIR):
    lane_lines = Video_Pipeline(max_counter=11)  # calling the class and getting the output Image
    leftupper = (585, 460)
    rightupper = (705, 460)
    leftlower = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
    # Defining the area
    warped_leftupper = (250, 0)
    warped_rightupper = (250, img.shape[0])
    warped_leftlower = (1050, 0)
    warped_rightlower = (1050, img.shape[0])
    # Defining the src and dst values for prespective transformation
    src = np.float32([leftupper, leftlower, rightupper, rightlower])
    dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])
    # Getting the video input frame and running the pipeline and saving to output directory
    lane_lines.set_presp_indices(src, dst)
    image_clip = VideoFileClip(input_video_file)
    white_clip = image_clip.fl_image(lane_lines.process_image)
    white_clip.write_videofile(output_dir + input_video_file, audio=False)


########################################################################################################################

# print("Output Video Generated Successfully ! ", create_video_output(PRJCT_VIDEO))

def chooose_input_video():
    choice = input("Enter a Number to run the pipeline for the video! \n"
                   "1 Run pipeline on Project Video\n"
                   "2 Run pipeline on Challenge Video\n"
                   "3 Run pipeline on Hard Challenge Video\n"
                   "4 Exit \n")
    if choice == "1":
        print("Output Video Generated Successfully ! ", create_video_output(PRJCT_VIDEO))
    elif choice == "2":
        print("Output Video Generated Successfully ! ", create_video_output(CHLNG_VIDEO))
    elif choice == "3":
        print("Output Video Generated Successfully ! ", create_video_output(HARDC_VIDEO))
    elif choice == 4:
        print("Exiting the Code", exit(0))
    else:
        print("Exiting the Code", exit(0))


chooose_input_video()
########################################################################################################################
