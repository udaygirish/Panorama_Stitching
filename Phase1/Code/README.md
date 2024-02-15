# P1:MyAutoPano - Phase 1 (Traditional Approach)

Course Project 1 for RBE549 - Computer Vision (Spring 2024)

Team Members: UdayGirish Maradana, Pradnya Sushil Shinde

## Requirements

1. CUDA Toolkit + GPU drivers

2. Pytorch

3. Numpy

4. Matplotlib

5. Opencv

6. Scikit-Image

### Implementation 

1. To stitch a given set of images, run the following: 
```
python3 Wrapper.py --NumFeatures <number_of_features> --InputPath <path_to_data> --ClearResults <Yes/No> 

```

Summary:

Stitches a given set of images (Train/Test) with a procedural flow of: Corner Detection, Adaptive Non Maximal Suppression, Feature Description, Feature Matching, Outlier Rejection using RANSAC, Image Warping and Blending

Saves the output of every procedure in the corresponding folders.