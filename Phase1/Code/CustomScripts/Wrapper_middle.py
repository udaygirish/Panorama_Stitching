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

    left_img_list = [int(i.split(".")[0]) for i in img_list[: len(img_list) // 2]]
    right_img_list = [int(i.split(".")[0]) for i in img_list[len(img_list) // 2 :]]

    # Center to left
    left_img_list = left_img_list[::-1]
    right_img_list = right_img_list[::-1]

    print(left_img_list)
    print(right_img_list)

    # Perform left iteration
    temp_out = imgs[left_img_list[0] - 1]
    for i in tqdm(range(1, len(left_img_list))):
        img2 = imgs[left_img_list[i] - 1]
        # temp_out = cv2.resize(temp_out, (640, 480))
        # img2 = cv2.resize(img2, (640, 480))
        temp_out = stitcher.stitch_pair(temp_out, img2)
        file_handler.write_output(temp_out, "Left/", "left{}.png".format(i))

    left_temp_out = temp_out

    # Perform right iteration
    temp_out = imgs[right_img_list[0] - 1]
    for i in tqdm(range(1, len(right_img_list))):
        img2 = imgs[right_img_list[i] - 1]
        # temp_out = cv2.resize(temp_out, (640, 480))
        # img2 = cv2.resize(img2, (640, 480))
        temp_out = stitcher.stitch_pair(temp_out, img2)
        file_handler.write_output(temp_out, "Right/", "right{}.png".format(i))

    right_temp_out = temp_out

    # Perform final stitching
    final_out = stitcher.stitch_pair(left_temp_out, right_temp_out, threshold=0)
    file_handler.write_output(final_out, "Final/", "final.png")


if __name__ == "__main__":
    main()
