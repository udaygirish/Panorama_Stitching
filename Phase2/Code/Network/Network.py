"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl
from kornia.geometry.transform.imgwarp import get_perspective_transform
from kornia.geometry.transform import warp_perspective
import cv2
from kornia.geometry.homography import find_homography_dlt
import torch.nn.init as init
from Network.transformer import spatial_transformer
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SLossFn(delta, gt):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    # Assuming delta and gt are both torch tensors
    # Reshape delta and gt to have the same shape
    # delta = delta.view(-1, 4, 2)
    # gt = gt.view(-1, 4, 2)

    # Calculate the L2 loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(delta, gt)

    return loss


def SLoss_Root(delta, gt):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(delta, gt)
    loss = torch.sqrt(loss)
    return loss


def UnSLossFn(pred_patch_b, patch_b):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    # Assuming img_a and patch_b , corners are all torch tensors
    # SSIM is a similarity loss from Kornia library
    # Have to check which is better L1 or SSIM
    # loss_fn = kornia.losses.ssim_loss(pred_patch_b, patch_b, 5, reduction="mean")
    # loss = loss_fn(pred_patch_b, patch_b)
    # L1 Loss
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(pred_patch_b, patch_b)
    return loss


def SSIMLoss(pred_patch_b, patch_b):
    loss = kornia.losses.ssim_loss(pred_patch_b, patch_b, 5, reduction="mean")
    return loss


class HomographyModel(nn.Module):
    def training_step(self, batch):
        patch, gt = batch
        delta = self(patch)
        loss = SLossFn(delta, gt)
        # print("Delta: ", delta)
        # print("GT: ", gt)
        return {"loss": loss}

    def validation_step(self, batch):
        patch, gt = batch
        delta = self(patch)
        loss = SLossFn(delta, gt)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Supervised_Net(HomographyModel):
    def __init__(self, InputSize, OutputSize, type="regressor"):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Supervised_Net, self).__init__()
        self.model_type = type
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 16 * 128, 1024), nn.ReLU(), nn.Linear(1024, 8)
        )

    def forward(self, xa):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        # Append the two images together depth wise
        # Revert the order of the channels to 2 x 128 x 128
        # input = input.permute(1, 0, 2, 3)
        out = self.conv_layers(xa)
        out = out.view(-1, 16 * 16 * 128)
        out = self.fc_layers(out)
        # Reshape the output to 4 x 2 using view
        # out = out.view(-1, 4, 2)

        return out


class UnHomographyModel(nn.Module):
    def training_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta, h4pt, h3m = self(patch, patcha, imgA, corners)
        loss = UnSLossFn(delta, patchb)
        return {"loss": loss}

    def validation_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta, h4pt, h3m = self(patch, patcha, imgA, corners)
        loss = UnSLossFn(delta, patchb)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Unsupervised_Net(UnHomographyModel):
    def __init__(self, InputSize, OutputSize, type="regressor"):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Unsupervised_Net, self).__init__()
        self.model_type = type
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 16 * 128, 1024), nn.ReLU(), nn.Linear(1024, 8)
        )

        # Different weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Initialize the weights/bias with identity transformation
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(
        #     torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        # )

    # def construct_A(src, dst):
    #     num_points = src.shape[0]
    #     matrices = []
    #     for i in range(num_points):
    #         x, y = src[i, 0], src[i, 1]
    #         u, v = dst[i, 0], dst[i, 1]
    #         matrices.append(
    #             [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
    #         )

    # Here total patch is patch, patcha, corners
    def forward(self, patch, patcha, imgA, corners):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        # Convert the total patch to a list
        out = self.conv_layers(patch)
        out = out.view(-1, 16 * 16 * 128)
        out = self.fc_layers(out)
        # Reshape the output to 4 x 2 using view
        h4pt = out.view(-1, 4, 2)
        # print(h4pt.shape)
        # Add the (-1,4,2) h4pt to the corners which ar e(-1,4,2)
        # h4pt = torch.mul(h4pt, 32)
        c4bpt = torch.add(corners, h4pt)

        # c4bpt = c4bpt - c4bpt[:, 0].view(-1, 1, 2)
        corners = corners - corners[:, 0].view(-1, 1, 2)
        # Multiply the corners and cb4pt by 128 using torch.mul
        # corners = torch.mul(corners, 128)
        # cb4pt = torch.mul(c4bpt, 128)
        # Convert every value in corners and cb4pt to int by floor
        # corners = torch.tensor(corners)
        # cb4pt = torch.tensor(cb4pt)
        # Get the perspective transform of batch of corners and cb4pt
        # Convert the c4bpt to float
        # Print the corners and c4bpt dtype and device
        h3m = find_homography_dlt(corners, c4bpt)
        # Warp the image
        h3m = torch.inverse(h3m)
        # Convert h3m to double
        h3m = h3m.double()
        imgA = imgA.double()
        # print(imgA.shape)
        patcha = patcha.double()  # * 255
        warped_patch = warp_perspective(
            imgA,
            h3m,
            dsize=(128, 128),
            padding_mode="zeros",
            mode="bilinear",
        )
        # print("====================================")
        # print("WARPED PATCH")
        # print(warped_patch)
        # print("H4PT")
        # print(h4pt)
        # print("H3M")
        # print(h3m)
        # print("CORNERS")
        # print(corners)
        # print("C4BPT")
        # print(c4bpt)

        return warped_patch, h4pt, h3m


def he_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias.data)


# New approach just by adding everything of TensorDLT and the Spatial loss to the loss function
def UnSLossFn1(h4pt, patcha, patchb, imgA, corners):
    # Use h4pt to find new corners
    corners_hat = corners + h4pt

    # Subtract the first corner from all corners
    corners = corners - corners[:, 0].view(-1, 1, 2)
    # Use corners and corners_hat to find the homography
    h3 = find_homography_dlt(corners, corners_hat)
    # Invert the homography
    h3 = torch.inverse(h3)
    h3 = h3.double()
    patcha = patcha.double()
    # Convert the homography to double
    patch_b_hat = warp_perspective(
        patcha, h3, dsize=(128, 128), padding_mode="zeros", mode="bilinear"
    )
    # Calculate the L1 loss
    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(patch_b_hat, patchb)
    return loss


class UnHomographyModel1(nn.Module):
    def training_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta, h4pt, h3m = self(patch, patcha, imgA, corners)
        loss = UnSLossFn(delta, patchb)
        return {"loss": loss}

    def validation_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta, h4pt, h3m = self(patch, patcha, imgA, corners)
        loss = UnSLossFn(delta, patchb)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Unsupervised_Net1(UnHomographyModel1):
    def __init__(self, InputSize, OutputSize, type="regressor"):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(Unsupervised_Net1, self).__init__()
        self.model_type = type
        #############################
        # Fill your network initialization of choice here!
        #############################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 16 * 128, 1024), nn.ReLU(), nn.Linear(1024, 8)
        )

    def tensor_dlt(self, corners, offset):
        corners = corners.view(-1, 8)
        bs, _ = corners.shape
        divide = int(np.sqrt(bs / 2) - 1)
        row_num = divide + 1
        row_num = row_num * 2

        corners_ = []
        pred_h4p_ = []

        for i in range(divide):
            for j in range(divide):
                h4p_indices = [
                    2 * j + row_num * i,
                    2 * j + row_num * i + 1,
                    2 * (j + 1) + row_num * i,
                    2 * (j + 1) + row_num * i + 1,
                    2 * (j + 1) + row_num * i + row_num,
                    2 * (j + 1) + row_num * i + row_num + 1,
                    2 * j + row_num * i + row_num,
                    2 * j + row_num * i + row_num + 1,
                ]

                h4p = corners[:, h4p_indices].reshape(bs, 1, 4, 2)
                pred_h4p = offset[:, h4p_indices].reshape(bs, 1, 4, 2)

                if i + j == 0:
                    corners_ = h4p
                    pred_h4p_ = pred_h4p
                else:
                    corners_ = torch.cat((corners_, h4p), 1)
                    pred_h4p_ = torch.cat((pred_h4p_, pred_h4p), 1)

        bs, n, h, w = corners_.shape
        N = bs * n
        corners_ = corners_.reshape(N, h, w)
        offset = offset.reshape(N, h, w)

        # Use h4p to find new corners
        corners_hat = corners_ + offset

        ones = torch.ones(N, 4, 1, device=corners.device)
        xy1 = torch.cat((corners_hat, ones), 2)
        zeros = torch.zeros_like(xy1)

        if torch.cuda.is_available():
            ones = ones.cuda()
            zeros = zeros.cuda()

        xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
        M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
        M2 = torch.matmul(
            corners_hat.reshape(-1, 2, 1), corners_.reshape(-1, 1, 2)
        ).reshape(N, -1, 2)

        A = torch.cat((M1, -M2), 2)
        b = corners_.reshape(N, -1, 1)

        Ainv = torch.inverse(A)
        # H8 matrix - H9 (1)
        h8 = torch.matmul(Ainv, b).reshape(N, 8)

        H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, n, 3, 3)
        return H

    # Spatial Transformer
    def transform(
        self,
        patch_size,
        M_tile_inv,
        H_mat,
        M_tile,
        I,
        patch_indices,
        batch_indices_tensor,
    ):
        # Transform H_Mat since we scale image indicies in transformer
        batch_size, num_channels, height, width = I.size()
        # Hmat
        # Drop third axis Hmat (bs,1,3,3) -> (bs,3,3)
        H_mat = H_mat.squeeze(1)
        M_tile = M_tile.double()
        M_tile_inv = M_tile_inv.double()
        H_mat = H_mat.double()
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)
        out_size = (height, width)
        warped_images, _ = spatial_transformer(I, H_mat, out_size)
        warped_images = warped_images.permute(0, 2, 3, 1)
        warped_gray_images = torch.mean(warped_images, dim=3)
        warped_images_flat = torch.reshape(warped_gray_images, [-1])
        patch_indices_flat = torch.reshape(patch_indices, [-1])
        pixel_indices = patch_indices_flat.long() + batch_indices_tensor
        pred_I2_flat = torch.gather(warped_images_flat, 0, pixel_indices)
        pred_I2 = torch.reshape(pred_I2_flat, [batch_size, patch_size, patch_size, 1])

        return pred_I2.permute(0, 3, 1, 2)

    # Here total patch is patch, patcha, corners
    def forward(self, patch, patcha, imgA, corners):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        warped_patch = None
        h4pt = None
        h3m = None
        batch_size, num_channels, height, width = imgA.size()
        _, _, patch_size, _ = patch.size()

        y_t = torch.arange(0, batch_size * width * height, width * height)
        batch_indices_tensor = (
            y_t.unsqueeze(1).expand(batch_size, patch_size * patch_size).reshape(-1)
        )
        M_tensor = torch.tensor(
            [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]]
        )

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()
            batch_indices_tensor = batch_indices_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(
            batch_size, M_tensor.shape[-2], M_tensor.shape[-1]
        )
        # Inverse of M

        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(
            batch_size, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1]
        )

        conv_out = self.conv_layers(patch)
        conv_out = conv_out.view(-1, 16 * 16 * 128)
        fc_out = self.fc_layers(conv_out)
        H_mat = self.tensor_dlt(corners, fc_out)
        warped_patch = self.transform(
            patch_size, M_tile_inv, H_mat, M_tile, imgA, patcha, batch_indices_tensor
        )

        return warped_patch, fc_out, H_mat
