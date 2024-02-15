# Import required libraries
import torch
import torchvision
from torchvision import transforms, datasets
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


def SGenerateBatch(generate_batch_info, MiniBatchSize):
    # We need - patch_a, patch_b , gt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    I1Batch = []
    # {"patchA": patchATrain, "patchB": patchBTrain, "patchAlist": patchAtrainlist, "patchBlist": patchBtrainlist}
    patchA = generate_batch_info["patchA"]
    patchB = generate_batch_info["patchB"]
    patchAlist = generate_batch_info["patchAlist"]
    patchBlist = generate_batch_info["patchBlist"]
    # print("First Image in Patch A: ", patchAlist[0])
    # print("First Image in Patch B: ", patchBlist[0])
    # print("Number of Images in Patch A: ", len(patchAlist))
    # print("Number of Images in Patch B: ", len(patchBlist))

    ImageNum = 0
    patchAbatch = []
    patchBbatch = []
    patchbatch = []
    homographybatch = []
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(patchAlist) - 1)
        PatchAName = patchA + patchAlist[RandIdx]
        PatchBName = patchB + patchBlist[RandIdx]
        # print(PatchAName)
        img_key_id = PatchAName.split("/")[-1].split(".")[0]
        # print(img_key_id)
        ImageNum += 1

        imgA = cv2.imread(PatchAName)
        imgB = cv2.imread(PatchBName)

        # Form a 2 channel image
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        # Convert to float32
        imgA = np.float32(imgA)
        imgB = np.float32(imgB)

        imgA = imgA / 255
        imgB = imgB / 255

        img = np.dstack((imgA, imgB))
        # imgA = torch.from_numpy(imgA)
        # imgB = torch.from_numpy(imgB)

        # Normalize
        # imgA = (imgA - torch.mean(imgA)) / torch.std(imgA)
        # imgB = (imgB - torch.mean(imgB)) / torch.std(imgB)

        # Stack the images depth wise to form a 2 channel image
        # img = torch.stack((imgA, imgB), dim=2)

        img = torch.from_numpy(img)

        homography = generate_batch_info["homography"][str(img_key_id)]
        # print(len(generate_batch_info["homography"]))
        # print(patchAlist[RandIdx])
        # print(patchAlist[RandIdx])
        # print(RandIdx)
        homography = torch.from_numpy(homography)
        # Convert 4*2 to 8*1
        homography = homography.view(-1, 8)
        # Reshape from 1*8 to 8
        homography = homography.squeeze(0)
        # Normalize
        homography = homography

        # Append All Images and Mas

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        patchAbatch.append(imgA)
        patchBbatch.append(imgB)
        img = img.permute(2, 0, 1)
        patchbatch.append(img)
        homographybatch.append(homography)

    # Convert the patchA and patchB batch to tensors by adding a batch dimension
    # patchAbatch = torch.stack(patchAbatch)
    # patchBbatch = torch.stack(patchBbatch)
    homographybatch = torch.stack(homographybatch)
    patchbatch = torch.stack(patchbatch)

    # patchAbatch = patchAbatch.to(device)
    # patchBbatch = patchBbatch.to(device)
    # homographybatch = homographybatch.to(device)

    # I1Batch.append(patchAbatch)
    # I1Batch.append(patchBbatch)
    I1Batch.append(homographybatch)

    return patchbatch, homographybatch


def infer_model(model, patchbatch):
    # Set model to evaluation mode
    model.eval()
    # Generate a batch of data
    output = model(patchbatch)
    return output


def get_EPE(output, homographybatch):
    # Calculate the EPE loss
    EPE_loss = torch.nn.MSELoss()
    # Calculate the EPE loss
    EPE = EPE_loss(output, homographybatch)
    # EPE = torch.sqrt(EPE)
    return EPE


def main():
    average_epe = []
    patch_base_path = "../Data/GeneratedTrainDataResized/"
    labels_path = "./TxtFiles/"
    homographybatch = np.load(
        labels_path + "homography_val_labels_resized.npy", allow_pickle=True
    ).item()
    prev_base_key = None
    batchsize = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patchA_list = os.listdir(patch_base_path + "patchA/")
    patchB_list = os.listdir(patch_base_path + "patchB/")
    total_time = []

    # Sort the patchA_list where the image key is of the form 1_1
    patchA_list.sort(key=lambda x: int(x.split("_")[0]))
    patchB_list.sort(key=lambda x: int(x.split("_")[0]))
    for i in tqdm(range(int(len(patchA_list) / batchsize))):
        # print("Current Batch: ", i)
        # print("Total Batches: ", int(len(patchA_list) / batchsize))
        temp_patch = []
        temp_homography = []
        count = 0
        for j in range(i * batchsize, len(patchA_list)):
            if count == batchsize:
                break
            count += 1
            patchA = patch_base_path + "patchA/" + patchA_list[j]
            patchB = patch_base_path + "patchB/" + patchB_list[j]
            img_key_id = patchA.split("/")[-1].split(".")[0]
            homography = homographybatch[img_key_id]

            imgA = cv2.imread(patchA)
            imgB = cv2.imread(patchB)

            # Form a 2 channel image
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

            # Convert to float32
            imgA = np.float32(imgA)
            imgB = np.float32(imgB)

            imgA = imgA  # 255
            imgB = imgB  # / 255

            img = np.dstack((imgA, imgB))
            img = torch.from_numpy(img)
            homography = torch.from_numpy(homography)
            homography = homography.view(-1, 8)
            homography = homography.squeeze(0)

            # permute img
            img = img.permute(2, 0, 1)
            temp_patch.append(img)
            temp_homography.append(homography)

        temp_patch = torch.stack(temp_patch)
        temp_homography = torch.stack(temp_homography)

        temp_patch = temp_patch.to(device)
        temp_homography = temp_homography.to(device)

        model = Supervised_Net((2, 128, 128), (8, 1))
        model.load_state_dict(
            torch.load("../Checkpoints/Nonorm2/84model.ckpt")["model_state_dict"]
        )
        model.to(device)
        temp1 = time.time()
        output = infer_model(model, temp_patch)
        temp2 = time.time()

        print("Time taken for inference: ", temp2 - temp1)
        print("Time taken for inference of 1 image: ", (temp2 - temp1) / batchsize)
        # Multiplication with 32 to transform to real coordinates
        output = output  # * 32

        epe = get_EPE(output, temp_homography)
        print(epe)
        average_epe.append(epe.item())

    print("Average EPE: ", sum(average_epe) / len(average_epe))


if __name__ == "__main__":
    main()
