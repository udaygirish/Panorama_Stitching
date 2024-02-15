import cv2
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np


# Add any python libraries here
class FileHandler:
    def __init__(self, base_path="Results/"):
        """
        Initialize the FileHandler class.

        Parameters:
        - base_path (str): The base path of the project.
        """
        self.base_path = base_path

    def check_folder_exists(self, folder_path):
        """
        Checks if a folder exists at the specified path, and creates it if it doesn't exist.

        Parameters:
        - folder_path (str): The path of the folder to be checked.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def write_output(self, image, method, name):
        """
        Writes the output image to the base path.

        Parameters:
        - image: The image to be written.
        - type_of_filter (str): The type of filter applied to the image.
        - name (str): The name of the output file.
        """
        self.check_folder_exists(self.base_path + method)
        cv2.imwrite(
            self.base_path + method + name,
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

    def plot_images(self, fig_size, filters, x_len, y_len, method, name):
        """
        Plots a grid of images and saves it as a file.

        Parameters:
        - fig_size (tuple): The size of the figure (width, height).
        - filters (list): A list of images to be plotted.
        - x_len (int): The number of columns in the grid.
        - y_len (int): The number of rows in the grid.
        - name (str): The name of the output file.
        """
        fig = plt.figure(figsize=fig_size)
        length = len(filters)
        for idx in np.arange(length):
            ax = fig.add_subplot(y_len, x_len, idx + 1, xticks=[], yticks=[])
            plt.imshow(filters[idx], cmap="gray")
        plt.axis("off")
        plt.savefig(self.base_path + method + name, bbox_inches="tight", pad_inches=0.3)
        plt.close()

    # ToDO: Fix this Issue
    def plot_images_cv2(self, fig_size, filters, x_len, y_len, name):
        """
        Plots a grid of images and saves it as a file.

        Parameters:
        - fig_size (tuple): The size of the figure (width, height).
        - filters (list): A list of images to be plotted.
        - x_len (int): The number of columns in the grid.
        - y_len (int): The number of rows in the grid.
        - name (str): The name of the output file.
        """
        # Create a blank image
        total_width = x_len * fig_size[0]
        total_height = y_len * fig_size[1]
        output_image = np.zeros((total_height, total_width), dtype=np.uint8)

        length = len(filters)
        for idx in np.arange(length):
            # Calculate the position of the current filter in the grid
            row = idx // x_len
            col = idx % x_len

            # Calculate the region where the current filter will be placed
            y_start = row * fig_size[1]
            y_end = y_start + fig_size[1]
            x_start = col * fig_size[0]
            x_end = x_start + fig_size[0]

            # Resize the filter image to match the specified size
            filter_image = cv2.resize(filters[idx], (fig_size[0], fig_size[1]))

            # Place the filter image in the output image
            output_image[y_start:y_end, x_start:x_end] = filter_image

        # Save the resulting image
        cv2.imwrite(name, output_image)
