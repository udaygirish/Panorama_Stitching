#### Note: PLEASE EXECUTE ALL THE SCRIPTS IN THE INDIVIDUAL PHASE CODES FOLDER

# Phase 1

## P1:MyAutoPano - Phase 1 (Traditional Approach)

Course Project 1 for RBE549 - Computer Vision (Spring 2024)

Team Members: UdayGirish Maradana, Pradnya Sushil Shinde

### Requirements

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

## P1:MyAutoPano - Phase 2 (Deep Learning Approach)

Course Project 1 for RBE549 - Computer Vision (Spring 2024)

Team Members: UdayGirish Maradana, Pradnya Sushil Shinde

### Requirements

1. CUDA Toolkit + GPU drivers

2. Pytorch

3. Numpy

4. Matplotlib

5. Opencv

6. Scikit-Image

### Implementation 

1. To train, run the following:
```
python3 Train.py --BasePath <base_path> --CheckPointPath <checkpoint_path> --ModelType <model_type> --NumEpochs <num_epochs> --DivTrain <div_train> --MiniBatchSize <minibatchsize> --LoadCheckPoint <0/1> --LogsPath <log_path>

```
Trains the model with Supervised or Unsupervised approach and given parameters and saves the corresponding checkpoints.

2. To test, run the following:
```
python3 Wrapper.py --DataSet <data_Set> --ModelType <model_type> --PathToData <path_to_data> --ImageSelection <image_selection> --ImageNum <image_num> --ImageCount <image_count>

```
Tests the model for Train, Val or Test Datasets by loading the checkpoints of selected model type (Supervised or Unsupervised) and saves  the image outputs in respective folders.




