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

# To Clear previous outputs

# Understand if morphological operations are needed
from scipy.ndimage.morphology import (
    generate_binary_structure,
    binary_erosion,
    binary_dilation,
)

from helpers.utils import FileHandler
from helpers.logger import setup_logger


class Stitcher:
    def __init__(self):
        """
        Initialize the Stitcher object.

        This class is used for panorama stitching of images.

        Attributes:
        - description (str): Description of the Stitcher class.
        - corner_threshold (float): Threshold for corner detection.
        - N_best (int): Number of best features to select.
        - patch_size (int): Size of the patch for feature matching.
        - ratio (float): Ratio for feature matching.
        - threshold (float): Homography threshold.
        - num_iterations (int): Number of iterations for homography estimation.
        - inlier_percent_threshold (float): Inlier percentage threshold for homography estimation.
        - logger (Logger): Logger object for logging.

        """
        self.description = "Stitcher Class for Panorama Stitching of the images"
        self.corner_threshold = 0.001
        self.N_best = 1000
        self.patch_size = 41
        self.ratio = 0.7
        self.threshold = 0.5  # Homography threshold
        self.num_iterations = 2000  # Number of iterations for Homography
        self.inlier_percent_threshold = 90
        self.logger = setup_logger()
        self.inlier_count_threshold = 20

    def get_corners(self, img_orig):
        """
        Detect corners in the image using Harris Corner Detection
        :param img_orig: Input image (Expects in BGR format)
        :return: Tuple containing the list of corner coordinates, corner scores, and the image with highlighted corners
        """
        img = img_orig.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        # Check for need of dilation or not
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        c_pts = np.where(dst > self.corner_threshold * dst.max())

        dst[dst <= self.corner_threshold * dst.max()] = 0
        c_scores = dst
        img[dst > self.corner_threshold * dst.max()] = [0, 0, 255]
        c_img = img
        # Log length of the c_pts and c_scores
        self.logger.info("Length of c_pts: {}".format(len(c_pts)))
        self.logger.info("Length of c_scores: {}".format(len(c_scores)))
        return c_pts, c_scores, c_img

    def compute_ANMS(self, c_scores):
        # Let us understand why we use this why not peak max scores
        local_maxima = maximum_filter(c_scores, 10)
        mask = c_scores == local_maxima

        local_minima = minimum_filter(c_scores, 10)
        self.logger.info("Local maxima max: {}".format(local_maxima.max()))
        diff = (local_maxima - local_minima) > 0.01 * local_maxima.max()
        # diff = local_maxima > 0.001 * local_maxima.max()
        mask[diff == 0] = 0

        # Get the coordinates of the local maxima
        coords = []
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j] == True:
                    coords.append([i, j])
        coords = np.array(coords)
        self.logger.info("Length of coords: {}".format(len(coords)))

        Nstrong = len(coords) - 1
        self.logger.info("Length of Nstrong: {}".format(Nstrong))

        # Compute distances
        ri = np.full(Nstrong, np.inf)
        for i in tqdm(range(Nstrong)):
            for j in range(Nstrong):
                if (
                    c_scores[coords[j][0], coords[j][1]]
                    > c_scores[coords[i][0], coords[i][1]]
                ):
                    dist = np.sqrt(
                        (coords[j][0] - coords[i][0]) ** 2
                        + (coords[j][1] - coords[i][1]) ** 2
                    )
                    if dist < ri[i]:
                        ri[i] = dist

        # Sort the distances in descending order
        sorted_indices = np.argsort(ri)[::-1]
        selected_indices = sorted_indices[: self.N_best]
        self.logger.info("Length of selected_indices: {}".format(len(selected_indices)))

        # Get the coordinates of the selected points
        selected_pts = np.zeros((self.N_best, 2))
        for i in range(self.N_best):
            selected_pts[i] = coords[selected_indices[i]]
        selected_pts = selected_pts.astype(int)
        self.logger.info("Length of selected_pts: {}".format(len(selected_pts)))
        return selected_pts

    def compute_ANMS_pkmax(self, c_scores):
        # Compute peak local maxima
        coords = peak_local_max(c_scores, min_distance=2, threshold_abs=0.01)
        self.logger.info("Length of coords: {}".format(len(coords)))

        Nstrong = len(coords) - 1
        self.logger.info("Length of Nstrong: {}".format(Nstrong))

        # Compute distances
        ri = np.full(Nstrong, np.inf)
        for i in tqdm(range(Nstrong)):
            for j in range(Nstrong):
                if (
                    c_scores[coords[j][0], coords[j][1]]
                    > c_scores[coords[i][0], coords[i][1]]
                ):
                    dist = np.sqrt(
                        (coords[j][0] - coords[i][0]) ** 2
                        + (coords[j][1] - coords[i][1]) ** 2
                    )
                    if dist < ri[i]:
                        ri[i] = dist

        # Sort the distances in descending order
        sorted_indices = np.argsort(ri)[::-1]
        selected_indices = sorted_indices[: self.N_best]
        self.logger.info("Length of selected_indices: {}".format(len(selected_indices)))

        self.N_best = min(self.N_best, len(coords))
        # Get the coordinates of the selected points
        selected_pts = np.zeros((self.N_best, 2))
        for i in range(self.N_best):
            selected_pts[i] = coords[selected_indices[i]]
        selected_pts = selected_pts.astype(int)
        self.logger.info("Length of selected_pts: {}".format(len(selected_pts)))
        return selected_pts

    def compute_feature_descriptors(self, img, c_pts):
        """
        Compute Feature Descriptors
        :param img: Input Image
        :param c_pts: Corner Points
        :return: Feature Descriptors
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        # Compute a patch size of 41x41
        factor = int(self.patch_size / 2)
        feat_vectors = []
        coords = []
        feature_patches = []
        for point in c_pts:
            x, y = point
            if (
                x > factor
                and y > factor
                and x < gray_img.shape[0] - factor
                and y < gray_img.shape[1] - factor
            ):
                patch = gray_img[
                    x - factor : x + factor + 1, y - factor : y + factor + 1
                ]
                patch = cv2.GaussianBlur(patch, (7, 7), cv2.BORDER_DEFAULT)
                subsampled_patch = cv2.resize(patch, (8, 8))
                feat_vector = subsampled_patch.reshape(64)
                std_feat_vect = (feat_vector - np.mean(feat_vector)) / np.std(
                    feat_vector
                )
                feat_vectors.append(std_feat_vect)
                coords.append([x, y])
                feature_patches.append(subsampled_patch)
        feat_vectors = np.array(feat_vectors)
        # coords = np.array(coords)
        return feat_vectors, coords, feature_patches

    def match_features(self, feat_vect1, feat_vect2, coords1, coords2):
        # Match features
        feature_matches = []
        for i in tqdm(range(len(feat_vect1))):
            sq_dist = []
            for j in range(len(feat_vect2)):
                dist = np.sum((feat_vect1[i] - feat_vect2[j]) ** 2)
                sq_dist.append(dist)

            sorted_indices = np.argsort(sq_dist)
            if sq_dist[sorted_indices[0]] / sq_dist[sorted_indices[1]] < self.ratio:
                feature_matches.append((coords1[i], coords2[sorted_indices[0]]))
        self.logger.info("Length of feature_matches: {}".format(len(feature_matches)))
        return feature_matches

    # def plot_best_matches(f)

    # RANSAC and Homography
    def compute_homography(self, matches):
        best_inliers = []
        best_H = None
        num_matches = len(matches)
        required_inliers = int(self.inlier_percent_threshold / 100 * num_matches)
        new_matches = []
        for match in matches:
            new_matches.append((match[0][::-1], match[1][::-1]))
        matches = new_matches
        for _ in tqdm(range(self.num_iterations)):
            random_indices = np.random.choice(num_matches, 4, replace=False)
            src_keypoints = np.float32([matches[i][0] for i in random_indices]).reshape(
                -1, 1, 2
            )
            dst_keypoints = np.float32([matches[i][1] for i in random_indices]).reshape(
                -1, 1, 2
            )
            H = cv2.getPerspectiveTransform(
                src_keypoints.squeeze(), dst_keypoints.squeeze()
            )

            # Compute Inliers
            inliers = []
            for match in matches:
                temp_src = np.float32(match[0]).reshape(-1, 1, 2)
                temp_dst = np.float32(match[1]).reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(temp_src, H)
                ssd = np.sum((transformed_points - temp_dst) ** 2)
                if ssd < self.threshold:
                    inliers.append(match)

            # Keep the largest set of inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H

            # Check the stopping condition
            if len(best_inliers) >= required_inliers:
                break

        # Compute Inliers
        self.logger.info("Number of Inliers: {}".format(len(best_inliers)))

        # Have too
        # Recompute Homography using all the inliers
        # src_keypoints = np.float32([match[0] for match in best_inliers])
        # dst_keypoints = np.float32([match[1] for match in best_inliers])
        # best_H = cv2.getPerspectiveTransform(
        #     src_keypoints.reshape(-1, 1, 2), dst_keypoints.reshape(-1, 1, 2)
        # )
        return best_H, best_inliers

    def get_warped_image(self, img1, img2, best_H):
        # Warp Images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, best_H)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel())
        t = [-xmin, -ymin]
        translate_vector = np.array(
            [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]
        )  # translate
        result = cv2.warpPerspective(
            img2,
            translate_vector.dot(best_H),
            (xmax - xmin, ymax - ymin),
            flags=cv2.INTER_LINEAR,
        )

        result[t[1] : h1 + t[1], t[0] : w1 + t[0]] = img1
        # mask = np.zeros_like(img1, dtype=np.uint8)
        # cv2.fillConvexPoly(
        #     mask, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]]), (255, 255, 255)
        # )
        # result = cv2.seamlessClone(
        #     result, img1, mask, (w1 // 2, h1 // 2), cv2.NORMAL_CLONE
        # )
        # result = cv2.addWeighted(result, 0.5, img1, 0.5, 0)

        # Add Poisson Blending
        return result

    def detect_and_describe(self, image):
        # Use the above functions which takes a input image and gives the feature vectors
        # and the coordinates of the feature vectors and feature_patches
        corners, scores, c_img = self.get_corners(image)
        best_corners = self.compute_ANMS_pkmax(scores)
        feat_vect, coords, fp = self.compute_feature_descriptors(image, best_corners)
        return feat_vect, coords, fp

    def match_and_compute_homography(self, feat_vect1, feat_vect2, coords1, coords2):
        # Match features
        feature_matches = self.match_features(feat_vect1, feat_vect2, coords1, coords2)
        # Homography
        best_H, best_inliers = self.compute_homography(feature_matches)
        return feature_matches, best_H, best_inliers

    def stitch_pair(self, img1, img2, threshold=1):
        """
        Stitch a pair of images
        :param img1: Image 1
        :param img2: Image 2
        :return: Panorama
        """
        # Get corners
        c_pts1, c_scores1, c_img1 = self.get_corners(img1)
        self.logger.info("Corner Detection for Image 1 Done")
        c_pts2, c_scores2, c_img2 = self.get_corners(img2)
        self.logger.info("Corner Detection for Image 2 Done")

        # Get N_best corners
        best_c_pts1 = self.compute_ANMS_pkmax(c_scores1)
        self.logger.info("ANMS for Image 1 Done")
        best_c_pts2 = self.compute_ANMS_pkmax(c_scores2)
        self.logger.info("ANMS for Image 2 Done")

        # Compute feature descriptors
        feat_vect1, coords1, fp1 = self.compute_feature_descriptors(img1, best_c_pts1)
        self.logger.info("Feature Descriptor for Image 1 Done")
        feat_vect2, coords2, fp2 = self.compute_feature_descriptors(img2, best_c_pts2)
        self.logger.info("Feature Descriptor for Image 2 Done")

        # Match features
        feature_matches = self.match_features(feat_vect1, feat_vect2, coords1, coords2)
        self.logger.info("Feature Matching Done")

        # Compute Homography
        best_H, best_inliers = self.compute_homography(
            feature_matches,
        )
        self.logger.info("Homography Computed")
        self.logger.info("HOMOGRAPHY MATRIX: {}".format(best_H))

        # Warp images
        if threshold:
            if self.is_stitchable(len(best_inliers)):
                warped_image = self.get_warped_image(img2, img1, best_H)
                self.logger.info("Images Stitched Successfully")
            else:
                warped_image = img1
                self.logger.info("Images Not Stitchable")
        else:
            warped_image = self.get_warped_image(img2, img1, best_H)
            self.logger.info("Images Stitched Successfully")
        return warped_image

    def is_stitchable(self, inlier_count):
        if inlier_count > self.inlier_count_threshold:
            return True
