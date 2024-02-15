#!/usr/bin/evn python
import numpy as np
import cv2
import os
from tqdm import tqdm


def saveLabels(homography, corners, labels_path, type="train"):
    np.save(labels_path + "homography_{}_labels_resize.npy".format(type), homography)
    np.save(labels_path + "{}_corner_coordinates_resize.npy".format(type), corners)
    # print("Data Labels saved successfully....!")


def saveImgPatches(i, j, patch_A, patch_B, patches_path):
    save_patch_A = patches_path + "patchA/"
    save_patch_B = patches_path + "patchB/"

    if not os.path.isdir(save_patch_A) or not os.path.isdir(save_patch_B):
        os.makedirs(save_patch_A)
        os.makedirs(save_patch_B)

    cv2.imwrite(save_patch_A + str(i + 1) + "_" + str(j + 1) + ".jpg", patch_A)
    cv2.imwrite(save_patch_B + str(i + 1) + "_" + str(j + 1) + ".jpg", patch_B)
    # print("Image Patches saved successfully...!")


def generate_data(imgA, p_size, perturbation_range):
    # img_A = Input Image (h x w)
    # Patch A = random patch from img_A
    # Pertubaration range = Pixel shifts allowed

    h, w = imgA.shape[:2]

    # Define random top-left corner points for genrating Patch A
    x_min = np.random.randint(0, w - p_size)
    y_min = np.random.randint(0, h - p_size)

    # Define bottom-right corner points for generating Patch A
    x_max = x_min + p_size
    y_max = y_min + p_size

    # Define corner coordinates of Patch A
    # coords_A = np.array(
    #     [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    # )

    coords_A = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )

    # Allowed pixel shifts for computing pertubation
    # Outputs a 4 x 2 array with shifts of corner points in x and y directions
    perturbation = np.random.randint(
        -perturbation_range, perturbation_range + 1, (4, 2)
    )

    perturbed_coords_B = [
        [x + dx, y + dy] for (x, y), (dx, dy) in zip(coords_A, perturbation)
    ]
    # perturbed_coords_B = np.array(perturbed_coords_B)

    # Compute inverse homography between Image A coords and perturbed Image B coords
    H_inverse = np.linalg.inv(
        cv2.getPerspectiveTransform(
            np.float32(coords_A), np.float32(perturbed_coords_B)
        )
    )

    # Warp Image A with the inverse homography
    imgB = cv2.warpPerspective(imgA, H_inverse, (w, h))

    patch_A = imgA[y_min:y_max, x_min:x_max]
    patch_B = imgB[y_min:y_max, x_min:x_max]

    H4pt = (perturbed_coords_B - coords_A).astype(np.float32)

    # Calculate image corner coordinates as a tuple of corner coordinates of image A and the corresponding perturbed image B corner coordinates
    corner_coords = [(coords_A[i], perturbed_coords_B[i]) for i in range(len(coords_A))]

    return patch_A, patch_B, H4pt, corner_coords


def main():
    homography_vals = dict()
    corners = dict()
    p_size = 128
    perturbation_range = 32
    train_img_path = "../../Data/Val/"
    labels_path = "../TxtFiles/"
    patches_path = "../../Data/GeneratedValDataResize/"

    # print("Initiating data generation...")
    for i in tqdm(range(len(os.listdir(train_img_path)))):  #
        imgA = cv2.imread(train_img_path + str(i + 1) + ".jpg")
        # Resize the image to 640 (Width) x 480 (Height)
        imgA = cv2.resize(imgA, (640, 480))
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        # print("Computing data for image" + str(i + 1))
        for j in range(75):
            patch_A, patch_B, H4pt, corner_coords = generate_data(
                imgA, p_size, perturbation_range
            )
            saveImgPatches(i, j, patch_A, patch_B, patches_path)
            homography_vals[str(i + 1) + "_" + str(j + 1)] = H4pt
            corners[str(i + 1) + "_" + str(j + 1)] = corner_coords
        # print("Completed computing data for image" + str(i + 1))

    homography_vals = np.array(homography_vals)
    corners = np.array(corners)

    saveLabels(homography_vals, corners, labels_path, type="val")

    np_test = np.load(
        labels_path + "homography_val_labels_resize.npy", allow_pickle=True
    )
    # Print the shape of the loaded data
    print(np_test.shape)
    # Convert np array to dict
    np_test = np_test.item()
    print(len(np_test))
    print(np_test["1_1"])


# main()
