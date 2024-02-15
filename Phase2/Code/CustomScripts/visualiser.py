import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.environ["QT_LOGGING_RULES"] = "*.debug=false"

# Visualize images from two different folders
folder1 = "../Data/GeneratedTrainData1/patchA/"
folder2 = "../Data/GeneratedTrainData1/patchB/"

train_homography = np.load("./TxtFiles/homography_train_labels.npy")
print(len(train_homography))


# Generate random number between 0 and 5000
randomNumber = np.random.randint(0, 4)
randomNumber1 = np.random.randint(0, 4)

# Read image from folder1
img1 = cv2.imread(folder1 + str(randomNumber) + "_" + str(randomNumber1) + ".jpg")

# Read image from folder2
img2 = cv2.imread(folder2 + str(randomNumber) + "_" + str(randomNumber1) + ".jpg")

# # Plot images
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.show()

# Imshow in CV2 by appending both images with black border
# img = cv2.hconcat([img1, img2])
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(train_homography[randomNumber * randomNumber1].shape)
