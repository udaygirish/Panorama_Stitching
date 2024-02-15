import torch

# import torchvision
# from torchvision import transforms, datasets
import torch.nn as nn
from Network.Network import Supervised_Net, Unsupervised_Net
import cv2
import sys
import numpy as np
import random
import skimage
import PIL
import os
import glob
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
from termcolor import colored, cprint
import math
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def GenerateCustomData(imgA, p_size, perturbation_range):
    # img_A = Input Image (h x w)
    # Patch A = random patch from img_A
    # Pertubaration range = Pixel shifts allowed

    h, w = imgA.shape[:2]

    # Define random top-left corner points for generating Patch A
    x_min = np.random.randint(0, w - p_size)
    y_min = np.random.randint(0, h - p_size)

    # Define bottom-right corner points for generating Patch A
    x_max = x_min + p_size
    y_max = y_min + p_size

    # Define corner coordinates of Patch A
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


def get_EPE(output, homographybatch):
    # Calculate the EPE loss
    EPE_loss = torch.nn.MSELoss()
    # Calculate the EPE loss
    EPE = EPE_loss(output, homographybatch)
    # EPE = torch.sqrt(EPE)
    return EPE


def infer_model(
    model,
    patchbatch,
    pA,
    pB,
    test_img_tensor,
    coordsA,
    type="Supervised",
):
    # Set model to evaluation mode
    model.eval()
    # Generate a batch of data
    if type == "Supervised":
        output = model(patchbatch)
    else:
        # Add extra dimension for patch A
        pA = torch.from_numpy(np.array(pA))
        pA = pA.unsqueeze(0)
        pA = pA.unsqueeze(0)
        coordsA = torch.from_numpy(np.array(coordsA))
        coordsA = coordsA.unsqueeze(0)
        coordsA = coordsA
        # Conver to float
        pA = pA * 255
        coordsA = coordsA * 255
        pA = pA.float()
        coordsA = coordsA.float()
        # Convert every tensor to cuda
        pA = pA.to(device)
        coordsA = coordsA.to(device)
        test_img_tensor = test_img_tensor.to(device)
        patchbatch = patchbatch.to(device)
        warped_patch, output, _ = model(patchbatch, pA, test_img_tensor, coordsA)
        output = output.view(-1, 8)

    return output


def predictCorners(
    coordsA, patch, homography, pA, pB, test_img_tensor, model_type, img_num
):
    patch = patch.to(device)
    print(patch.shape)
    homography = homography.to(device)

    # model_type = "unsupervised"
    if model_type == "Supervised":
        model = Supervised_Net((2, 128, 128), (8, 1))
        model.load_state_dict(
            torch.load("../sqrt_checkpoints/Checkpoints/49model.ckpt")[
                "model_state_dict"
            ]
        )
    else:
        model = Unsupervised_Net((2, 128, 128), (8, 1))
        model.load_state_dict(
            torch.load("../Checkpoints/UnsupTest7/12model.ckpt")["model_state_dict"]
        )
    model.to(device)

    print(patch.shape)
    homography_pred = (
        infer_model(model, patch, pA, pB, test_img_tensor, coordsA, model_type) * 32
    )
    print(
        "Homography Predicted for Image Number " + str(img_num) + ": ", homography_pred
    )
    print("Homography for Image Number " + str(img_num) + ": ", homography)

    epe = get_EPE(homography_pred, homography)
    print("EPE for Image Number " + str(img_num) + ": ", round(epe.item(), 4))
    epe = round(epe.item(), 4)
    homography_pred = homography_pred.view(4, 2)
    homography_pred = homography_pred.cpu().detach().numpy()

    coordsB_pred = coordsA + homography_pred

    return coordsB_pred, epe


def saveImage(img, img_num, img_ct, dataset):
    save_img_path = "./OutputImages/" + dataset + "Images/"
    if not os.path.isdir(save_img_path):
        os.makedirs(save_img_path)

    # Clear Out Previous Iteration Results
    # if (img_ct == 1) and os.listdir(save_img_path): #If new image set is being tested and the directory is not empty
    #     shutil.rmtree(save_img_path)

    cv2.imwrite(save_img_path + str(img_num) + "_" + str(img_ct) + ".jpg", img)

    print("Output " + dataset + "Image " + str(img_num) + "_" + str(img_ct) + " saved!")


def visualize(coordsA, coordsB, coordsB_pred, epe, imageA, img_num, img_ct, dataset):
    coordinates = [coordsA, coordsB, coordsB_pred]
    colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0)]
    for e, element in enumerate(coordinates):
        color = colors[e]
        for i in range(len(element)):
            point1 = [int(coord) for coord in element[i]]
            pt1 = tuple(point1)
            point2 = [int(coord) for coord in element[(i + 1) % len(element)]]
            pt2 = tuple(point2)
            # print('Point 1', pt1)
            # print('Point 2', pt2)
            cv2.line(
                imageA,
                pt1,
                pt2,
                color,
                3,
            )

    text_size = cv2.getTextSize(
        "RMSE:" + str(round(math.sqrt(epe), 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    h, w = imageA.shape[:2]
    tx = 10
    ty = 30
    cv2.putText(
        imageA,
        "RMSE: " + str(round(math.sqrt(epe), 2)),
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )

    print("saving output images")
    saveImage(imageA, img_num, img_ct, dataset)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--DataSet", help="Choose from: Train, Test, Val", default="Train"
    )
    Parser.add_argument(
        "--ModelType",
        help="Choose from: Supervised, Unsupervised",
        default="Supervised",
    )
    Parser.add_argument(
        "--PathToData", help="Provide path to dataset.", default="../Data/Train/"
    )
    Parser.add_argument(
        "--ImageSelection",
        help="Choose whether to Randomize Selection of Images to Test: Random or Custom",
        default="Random",
    )
    Parser.add_argument(
        "--ImageNum",
        help="If Image Selection is custom, define an image number to start with.",
        default=1,
    )
    Parser.add_argument(
        "--ImageCount", help="Number of images from image number to test", default=1
    )

    # Delete Previous results arg
    Parser.add_argument(
        "--DeletePreviousResults", help="Delete Previous Results", default=False
    )

    Args = Parser.parse_args()
    DataSet = Args.DataSet
    model_type = Args.ModelType
    DataPath = Args.PathToData
    ImageNum = Args.ImageNum
    ImageCt = int(Args.ImageCount)
    ImageSelection = Args.ImageSelection
    Dpr = Args.DeletePreviousResults

    # Clear Output Folder
    saveOutputPath = "./OutputImages/" + DataSet + "Images/"
    if Dpr:
        if os.path.exists(saveOutputPath):
            shutil.rmtree(saveOutputPath)
        else:
            print("The directory does not exist")

    if ImageSelection == "Random":
        ImageNum = None

    epe_list = []
    for i in range(ImageCt):
        # Select a random image form the data folder
        if ImageSelection == "Random":
            j = random.randint(1, len(DataPath))

        elif ImageSelection == "Custom":
            j = ImageNum + i

        print(
            "Reading a "
            + ImageSelection
            + "image from "
            + DataSet
            + " data: Image Number "
            + str(j)
        )

        imgA_path = DataPath + str(j) + ".jpg"
        print("Selected Image path:", imgA_path)
        imgA = cv2.imread(imgA_path)

        # Generate Custom Data for the Selected Image
        p_size = 128
        perturbation_range = 32
        patch_A, patch_B, H4pt, corner_coords = GenerateCustomData(
            imgA, p_size, perturbation_range
        )

        # Assign Image Labels
        homography = np.array(H4pt)
        homography = torch.from_numpy(homography)
        homography = homography.view(-1, 8)
        homography = homography.squeeze(0)

        # Assign Corner Coordinates
        corner_coords = np.array(corner_coords)
        coordsA = [element[0] for element in corner_coords]
        coordsB = [element[1] for element in corner_coords]

        # Define Image Tensor

        img_tensor = np.array(imgA)
        img_tensor = torch.from_numpy(img_tensor)

        # Add axis and permute
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        imgA_tensor = np.array(imgA)
        imgA_tensor = torch.from_numpy(imgA_tensor)

        imgA_tensor = imgA_tensor.unsqueeze(0)
        imgA_tensor = imgA_tensor.permute(0, 3, 1, 2)

        patch_A = cv2.cvtColor(patch_A, cv2.COLOR_BGR2GRAY)
        patch_B = cv2.cvtColor(patch_B, cv2.COLOR_BGR2GRAY)

        pA = np.float32(patch_A)
        pB = np.float32(patch_B)

        # Add Divide by 255 for previous training
        pA = pA / 255
        pB = pB / 255

        patch = np.dstack((pA, pB))
        patch = torch.from_numpy(patch)
        patch = patch.permute(2, 0, 1)

        # Add patch dimension
        patch = patch.unsqueeze(0)

        print("Predicting corner coordinates")
        coordsB_pred, epe = predictCorners(
            coordsA, patch, homography, pA, pB, imgA_tensor, model_type, j
        )
        epe_list.append(epe)

        print("Visualizing corners")

        visualize(coordsA, coordsB, coordsB_pred, epe, imgA, j, i + 1, DataSet)
    print("Average EPE: ", sum(epe_list) / len(epe_list))


if __name__ == "__main__":
    # GenerateTestBatch()
    main()
