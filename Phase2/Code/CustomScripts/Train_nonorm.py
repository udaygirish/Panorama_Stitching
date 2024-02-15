#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW, SGD
from Network.Network import Supervised_Net, Unsupervised_Net
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.GenerateData import generate_data
from torch.nn.utils import clip_grad_norm_

# torch model summary
from torchsummary import summary
from torch.autograd.profiler import profile, record_function, ProfilerActivity


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

        # imgA = imgA / 255
        # imgB = imgB / 255

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


def UnSGenerateBatch(
    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    # Need - img_a, patch_a, patch_b, corners, gt

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        Patch1 = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        Patch2 = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        transform = ToTensor()
        Img = transform(Img)
        # Standardize the image using torch
        Img = (Img - torch.mean(Img)) / torch.std(Img)
        I1 = np.float32(cv2.imread(RandImageName))
        Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hparams = []
    if ModelType == "Sup":
        # Predict output with forward pass
        model = Supervised_Net((2, 128, 128), 8)  # 2 channel input, 8 channel output
    elif ModelType == "Unsup":
        model = Supervised_Net((2, 128, 128), 8)  # 2 channel input, 8 channel output
    else:
        print("Model type not recognized")
        sys.exit()

    # Print summary of model
    print("====" * 10)
    print("Model Summary")
    #
    summary(model, (2, 128, 128), device="cpu")

    model.to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = AdamW(model.parameters(), lr=0.001)
    Optimizer = SGD(model.parameters(), lr=0.00005, momentum=0.9)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=20, gamma=0.1)
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    # Set Directories
    if ModelType == "Sup":
        generate_batch_info = dict()
        DirNamesTrain = "GeneratedTrainData/"
        Base_Train = BasePath + os.sep + "Train/"
        Base_Val = BasePath + os.sep + "Val/"
        patchATrain = BasePath + os.sep + DirNamesTrain + "patchA/"
        patchBTrain = BasePath + os.sep + DirNamesTrain + "patchB/"
        patchAtrainlist = os.listdir(patchATrain)
        patchBtrainlist = os.listdir(patchBTrain)
        DirNamesVal = "GeneratedValData/"
        patchAVal = BasePath + os.sep + DirNamesVal + "patchA/"
        patchBVal = BasePath + os.sep + DirNamesVal + "patchB/"
        patchAvallist = os.listdir(patchBVal)
        patchBvallist = os.listdir(patchBVal)
        Base_Train_list = os.listdir(Base_Train)
        Base_Val_list = os.listdir(Base_Val)
        train_homography = np.load(
            "./TxtFiles/homography_train_labels.npy", allow_pickle=True
        )
        val_homography = np.load(
            "./TxtFiles/homography_val_labels.npy", allow_pickle=True
        )
        train_corner = np.load(
            "./TxtFiles/train_corner_coordinates.npy", allow_pickle=True
        )
        val_corner = np.load("./TxtFiles/val_corner_coordinates.npy", allow_pickle=True)
        generate_batch_info["train"] = {
            "patchA": patchATrain,
            "patchB": patchBTrain,
            "patchAlist": patchAtrainlist,
            "patchBlist": patchBtrainlist,
            "corners": train_corner.item(),
            "homography": train_homography.item(),
        }
        generate_batch_info["val"] = {
            "patchA": patchAVal,
            "patchB": patchBVal,
            "patchAlist": patchAvallist,
            "patchBlist": patchBvallist,
            "corners": val_corner.item(),
            "homography": val_homography.item(),
        }
        NumTrainSamples = len(patchAtrainlist)
    elif ModelType == "Unsup":
        print("Yet to fill")

    # model = model.to(device)
    # Add model graph to tensorboard
    Writer.add_graph(model, torch.rand(1, 2, 128, 128).to(device))
    train_loss = []
    val_loss = []
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        I1Batch = []
        I1ValBatch = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            if ModelType == "Sup":
                I1Batch = SGenerateBatch(generate_batch_info["train"], MiniBatchSize)
                I1ValBatch = SGenerateBatch(generate_batch_info["val"], MiniBatchSize)
            elif ModelType == "Unsup":
                I1Batch = UnSGenerateBatch(
                    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
                )
                I1ValBatch = UnSGenerateBatch(
                    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
                )

            I1Batch = [x.to(device) for x in I1Batch]
            I1ValBatch = [x.to(device) for x in I1ValBatch]

            # Convert Tensors to cuda type
            # I1Batch = [x.to(device) for x in I1Batch]
            LossThisBatch = model.training_step(I1Batch)["loss"]

            Optimizer.zero_grad()
            LossThisBatch.backward()
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            Optimizer.step()

            # # Save checkpoint every some SaveCheckPoint's iterations
            # if PerEpochCounter % SaveCheckPoint == 0:
            #     # Save the Model learnt in this epoch
            #     SaveName = (
            #         CheckPointPath
            #         + str(Epochs)
            #         + "a"
            #         + str(PerEpochCounter)
            #         + "model.ckpt"
            #     )

            #     torch.save(
            #         {
            #             "epoch": Epochs,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": Optimizer.state_dict(),
            #             "loss": LossThisBatch,
            #         },
            #         SaveName,
            #     )
            #     print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(I1ValBatch)
            # Tensorboard
            Writer.add_scalars(
                "LossEveryIter",
                {"train": LossThisBatch, "val": result["val_loss"]},
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
            )
            # print(LossThisBatch.item())
            train_loss.append(LossThisBatch.item())
            val_loss.append(result["val_loss"].item())
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
            # Detach tensors to free up memory
            # del I1Batch, I1ValBatch, LossThisBatch, result
            # torch.cuda.empty_cache()
        # Scheduler step
        scheduler.step()
        # Average loss over the epoch
        avg_train_loss = round(sum(train_loss) / len(train_loss), 4)
        avg_val_loss = round(sum(val_loss) / len(val_loss), 4)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            SaveName,
        )
        #                "loss": LossThisBatch,
        print("\n" + SaveName + " Model Saved...")

        # Tensorboard
        Writer.add_scalars(
            "LossPerEpoch", {"train": avg_train_loss, "val": avg_val_loss}, Epochs
        )
        Writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], Epochs)
        Writer.flush()
        print("Epoch: ", Epochs, "Train Loss: ", avg_train_loss)
        print("Epoch: ", Epochs, "Val Loss: ", avg_val_loss)


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="/home/udaygirish/Projects/WPI/computer_vision/project1/YourDirectoryID_p1/Phase2/Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = "49model"  # FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
