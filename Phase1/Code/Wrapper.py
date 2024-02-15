#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mp
from helpers.logger import setup_logger
import time
from tqdm import tqdm
from scipy.ndimage import maximum_filter, minimum_filter
import shutil
from skimage.feature import peak_local_max
from helpers.utils import FileHandler
from stitcher import Stitcher
import matplotlib.cm as cm

# To Clear previous outputs

# Understand if morphological operations are needed
from scipy.ndimage.morphology import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
)

# Dont generate pycache
import sys
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import random

sys.dont_write_bytecode = True

# Initializing custom logger
logger = setup_logger()


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--NumFeatures",
        default=100,
        help="Number of best features to extract from each image, Default:100",
    )
    Parser.add_argument(
        "--InputPath",
        default="../Data/Train/Set2/",
        help="Path to the directory containing the images",
    )

    # Clear Previous results argument - default set to Yes
    Parser.add_argument(
        "--ClearResults",
        default="Yes",
        help="Clear Previous results - Yes or No",
    )

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    InputPath = Args.InputPath

    base_path = "Results/" + InputPath.split("/")[-2] + "/"
    logger.info("Base Path for Saving Results: {}".format(base_path))

    if Args.ClearResults == "Yes":
        logger.info("Clearing Previous Results")
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
            logger.info("Removed previous results")

    # Initialize Python Objects
    file_handler = FileHandler(base_path)
    stitcher = Stitcher()

    """
    Read a set of images for Panorama stitching
    """
    img_list = os.listdir(InputPath)
    img_list.sort()
    # Read Images
    imgs = []
    for img in img_list:
        imgs.append(cv2.imread(InputPath + img))

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    # For now take the first and second image only
    # img1 = imgs[0]
    # img2 = imgs[1]
    logger.info("Number of Images: {}".format(len(imgs)))
    logger.info("Image List: {}".format(img_list))
    temp_out = imgs[0]
    for i in tqdm(range(1, len(imgs))):
        img1 = temp_out
        img2 = imgs[i]
        # Get corners
        c_pts1, c_scores1, c_img1 = stitcher.get_corners(img1)
        logger.info("Corner Detection for Image 1 Done")
        c_pts2, c_scores2, c_img2 = stitcher.get_corners(img2)
        logger.info("Corner Detection for Image 2 Done")
        # Save images
        if i == 1:
            file_handler.write_output(
                c_img1, "CornerDetection/", "corner{}.png".format(i)
            )
        else:
            file_handler.write_output(
                c_img1, "CornerDetection/", "corner{}{}.png".format((i), (i + 1))
            )

        file_handler.write_output(
            c_img2, "CornerDetection/", "corner{}.png".format(i + 1)
        )

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        # Get N_best corners
        anms_img1 = img1.copy()
        anms_img2 = img2.copy()
        best_c_pts1 = stitcher.compute_ANMS_pkmax(c_scores1)
        logger.info("ANMS for Image 1 Done")
        best_c_pts2 = stitcher.compute_ANMS_pkmax(c_scores2)
        logger.info("ANMS for Image 2 Done")
        # Save images
        for point1 in best_c_pts1:
            cv2.circle(anms_img1, (point1[1], point1[0]), 1, (0, 0, 255), -1)
        if i == 1:
            file_handler.write_output(anms_img1, "ANMS/", "anms{}.png".format(i))
        else:
            file_handler.write_output(
                anms_img1, "ANMS/", "anms{}{}.png".format((i), (i + 1))
            )
        for point2 in best_c_pts2:
            cv2.circle(anms_img2, (point2[1], point2[0]), 1, (0, 0, 255), -1)
        file_handler.write_output(anms_img2, "ANMS/", "anms{}.png".format(i + 1))

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        # Compute feature descriptors
        feat_vect1, coords1, fp1 = stitcher.compute_feature_descriptors(
            img1, best_c_pts1
        )
        feat_vect2, coords2, fp2 = stitcher.compute_feature_descriptors(
            img2, best_c_pts2
        )

        # Save feature patches
        # for i in range(50):
        #     # Append all feature patches as matplotlib plot in 10 * 5 and save fig
        file_handler.check_folder_exists(base_path + "FeatureDescriptor/")
        file_handler.plot_images(
            (10, 10), fp1[:100], 10, 10, "FeatureDescriptor/", "feature_patches1.png"
        )
        file_handler.plot_images(
            (10, 10), fp2[:100], 10, 10, "FeatureDescriptor/", "feature_patches2.png"
        )

        # Save images
        fd_img1 = img1.copy()
        fd_img2 = img2.copy()
        for point1 in coords1:
            cv2.circle(fd_img1, (point1[1], point1[0]), 1, (0, 0, 255), -1)
        if i == 1:
            file_handler.write_output(
                fd_img1, "FeatureDescriptor/", "fd{}.png".format(i)
            )
        else:
            file_handler.write_output(
                fd_img1, "FeatureDescriptor/", "fd{}{}.png".format((i), (i + 1))
            )
        for point2 in coords2:
            cv2.circle(fd_img2, (point2[1], point2[0]), 1, (0, 0, 255), -1)
        file_handler.write_output(
            fd_img2, "FeatureDescriptor/", "fd{}.png".format(i + 1)
        )

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        # Match features
        feature_matches = stitcher.match_features(
            feat_vect1, feat_vect2, coords1, coords2
        )
        logger.info("Feature Matching Done")
        # Save images
        try:
            match_img = np.concatenate((img1, img2), axis=1)
            for match in feature_matches:
                x1, y1 = match[0]
                x2, y2 = match[1]
                cv2.circle(match_img, (y1, x1), 1, (0, 0, 255), 1)  # Red border circle
                cv2.circle(
                    match_img, (y2 + img1.shape[1], x2), 1, (0, 0, 255), 1
                )  # Red border circle
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )  # Random color
                cv2.line(
                    match_img, (y1, x1), (y2 + img1.shape[1], x2), color, 1
                )  # Random color line
            file_handler.write_output(
                match_img, "FeatureMatching/", "matching{}.png".format(i)
            )
        except:
            pass
        """
        Refine: RANSAC, Estimate Homography
        """
        # Compute Homography
        best_H, best_inliers = stitcher.compute_homography(feature_matches)

        if not stitcher.is_stitchable(len(best_inliers)):
            logger.info("Homography Computation Failed")
            logger.info("Skipping Current Image")
            continue

        logger.info("Homography Computed")
        logger.info("HOMOGRAPHY MATRIX: {}".format(best_H))

        # best_H = compute_homography_cv2(feature_matches)
        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        # Warp images
        warped_image = stitcher.get_warped_image(img2, img1, best_H)
        file_handler.write_output(warped_image, "Panorama/", "mypano{}.png".format(i))
        temp_out = warped_image
        logger.info("Panorama Done")
        logger.info("======================================================")


if __name__ == "__main__":
    main()
