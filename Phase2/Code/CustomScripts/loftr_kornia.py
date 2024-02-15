import kornia.feature as KF
import os
import cv2
import kornia as K
from kornia_moons.viz import draw_LAF_matches
import torch
import matplotlib.pyplot as plt
from stitcher import Stitcher

# Load the images
base_path = "../Data/Phase2Pano/"
folder_path = "tower/"

imgs_list = os.listdir(base_path + folder_path)
imgs_list.sort()
print(imgs_list)


# Make a Panorama with loftr match

for i in range(len(imgs_list) - 1):
    img1 = K.io.load_image(
        base_path + folder_path + imgs_list[i], K.io.ImageLoadType.RGB32
    )[None, ...]
    img2 = K.io.load_image(
        base_path + folder_path + imgs_list[i + 1], K.io.ImageLoadType.RGB32
    )[None, ...]
    # Convert to grayscale
    img1 = K.color.rgb_to_grayscale(img1)
    img2 = K.color.rgb_to_grayscale(img2)
    matcher = KF.LoFTR(pretrained="indoor_new")
    input = {"image0": img1, "image1": img2}
    # To GPU
    # input = {k: v.to("cuda") for k, v in input.items()}
    # Use torch inference mode to use GPU
    with torch.inference_mode():
        correspondences = matcher(input)
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(
        mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    inliers = inliers > 0

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts0).view(1, -1, 2),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts0.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(mkpts1).view(1, -1, 2),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
            torch.ones(mkpts1.shape[0]).view(1, -1, 1),
        ),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1.0, 0.5, 1),
            "feature_color": (0.2, 0.5, 1),
            "vertical": False,
        },
        ax=ax,
    )
    print("Number of matches: ", mkpts0.shape[0])
    plt.savefig("loftr_matches.png")

# matcher = LoFTR(pretrained="outdoor")
# input = {"image0": img1, "image1": img2}
# correspondences_dict = matcher(input)
