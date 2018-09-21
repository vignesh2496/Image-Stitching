#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 23:21:04 2018

@author: vignesh
"""

from harris import harris_corner_detector 
from harris import plot_corners_on_image
from harris import np
from harris import cv2


def disp(I):
    I = np.uint8(I)
    cv2.imshow('image', I)
    cv2.waitKey(0)


# Stitching original image with warped image
def stitch(org, warp):
    stitched = np.zeros(org.shape, dtype = np.uint8)
    for i in range(org.shape[0]):
        for j in range(org.shape[1]):
            if not np.array_equal(warp[i, j], [0, 0, 0]):
                stitched[i, j, :] = warp[i, j, :]
            else:
                stitched[i, j, :] = org[i, j, :]
    return stitched


# Compute feature points
def compute_features(img_src, img_dst):
    
    # Features of source
    src_corners = harris_corner_detector(img_src, 3, 0.04, 5, True)
    src_corners = src_corners[np.where(src_corners[:, 2] > 0.03 * np.max(src_corners[:, 2]))]
    src_corner_img = plot_corners_on_image(img_src, src_corners, False)
    disp(src_corner_img)
    
    # Features of destination
    dst_corners = harris_corner_detector(img_dst, 3, 0.04, 5, True)
    dst_corners = dst_corners[np.where(dst_corners[:, 2] > 0.01 * np.max(dst_corners[:, 2]))]
    dst_corner_img = plot_corners_on_image(img_dst, dst_corners, False)
    disp(dst_corner_img)
    
    # Choose feature points in (row, col) format
    src_corners = np.array([[22, 259], [22, 272], [22, 287], [23, 300], [24, 311], [43, 261], [44, 273], [45, 289], [46, 302], [47, 314], [64, 262], [64, 276], [65, 293], [65, 305], [67, 316], [88, 269], [89, 294], [98, 270], [99, 294], [142, 270], [143, 301], [175, 273], [177, 305], [106, 327], [82, 324], [246, 347]])
    dst_corners = np.array([[151, 823], [159, 837], [167, 853], [175, 865], [180, 874], [175, 819], [181, 834], [189, 850], [195, 863], [201, 872], [198, 815], [203, 832], [211, 850], [216, 861], [221, 871], [225, 818], [235, 846], [238, 818], [246, 845], [285, 807], [291, 842], [322, 802], [324, 838], [259, 874], [236, 875], [388, 865]])
        
    # Plot them
    src_corner_img = plot_corners_on_image(img_src, src_corners, True)
    disp(src_corner_img)
    #cv2.imwrite('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Output_Images/build_left_harris_26_corners.jpg', src_corner_img)
    dst_corner_img = plot_corners_on_image(img_dst, dst_corners, True)
    disp(dst_corner_img)
    #cv2.imwrite('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Output_Images/build_right_harris_26_corners.jpg', dst_corner_img)
    return src_corners, dst_corners


# Pad destination image
def padded_image(img_dst):
    temp = np.zeros((570, 1160, 3), dtype = np.uint8)
    temp[150 : 411, 800 : 1148, :] = img_dst
    return temp      
    

# Find Homography matrix
def homography(src_corners, dst_corners):
    num_corners = src_corners.shape[0];
    A = np.zeros((2 * num_corners, 9))
    for i in range(num_corners):
        src_r = src_corners[i, 0]
        src_c = src_corners[i, 1]
        dst_r = dst_corners[i, 0]
        dst_c = dst_corners[i, 1]
        A[2 * i] = (0, 0, 0, -src_r, -src_c, -1, dst_c * src_r, dst_c * src_c, dst_c)
        A[2 * i + 1] = (src_r, src_c, 1, 0, 0, 0, -dst_r * src_r, -dst_r * src_c, -dst_r)
    U, L, V_t = np.linalg.svd(A)
    return V_t[8].reshape((3, 3)) / V_t[8][8]
    
  
# Perspective transformation
def perspective_transform(img_src, H, dst_shape):
    warp = np.zeros(dst_shape)
    for i in range(img_src.shape[0]):
        for j in range(img_src.shape[1]):
            i_transform, j_transform, z_transform = np.dot(H, [i, j, 1]) 
            # Convert to 2D
            i_transform = int(i_transform / z_transform)
            j_transform = int(j_transform / z_transform)
            if i_transform >= 0 and i_transform < dst_shape[0] and j_transform >= 0 and j_transform < dst_shape[1]:
                warp[i_transform, j_transform] = img_src[i, j]
    return warp
    
    
# Warp source image    
# if opencv is True, opencv libraries are used for the warp, else the user-defined functions are used
def warp_source(src_corners, dst_corners, dst_shape, opencv = False):
    if opencv:
        # Convert to (x, y) format for opencv
        num_corners = src_corners.shape[0]
        src_corners_opencv = src_corners.copy()
        dst_corners_opencv = dst_corners.copy() 
        for i in range(num_corners):
            src_corners_opencv[i, 0], src_corners_opencv[i, 1] = src_corners_opencv[i, 1], src_corners_opencv[i, 0]
            dst_corners_opencv[i, 0], dst_corners_opencv[i, 1] = dst_corners_opencv[i, 1], dst_corners_opencv[i, 0]   
        H, status = cv2.findHomography(src_corners_opencv, dst_corners_opencv)
        warp = cv2.warpPerspective(img_src, H, (dst_shape[1], dst_shape[0]))
    else:
        H = homography(src_corners, dst_corners)
        warp = perspective_transform(img_src, H, dst_shape)
        # Closing
        #warp = cv2.morphologyEx(warp, cv2.MORPH_CLOSE, np.ones((3, 3)))
        #cv2.imwrite('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Output_Images/warp_without_opencv_closed.jpg', warp)
    warp_with_corners = plot_corners_on_image(warp, dst_corners, True)
    disp(warp_with_corners)
    #cv2.imwrite('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Output_Images/warp_corners_with_opencv.jpg', warp_with_corners)
    return warp


img_src = cv2.imread('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Input_Images/build_left.jpg')
img_dst = cv2.imread('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Input_Images/build_right.jpg')
img_dst_pad = padded_image(img_dst)
src_corners, dst_corners = compute_features(img_src, img_dst_pad)
warp = warp_source(src_corners, dst_corners, img_dst_pad.shape, True)
stitched = stitch(img_dst_pad, warp)
stitched = plot_corners_on_image(stitched, dst_corners, True)
disp(stitched)
#cv2.imwrite('/home/vignesh/Desktop/Computer-Vision-Assignments/Assignment_2/Output_Images/build_stitched_with_opencv.jpg', stitched)
cv2.destroyAllWindows()