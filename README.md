# stacked-hourglass
Our modification of the stacked-hourglass architecture for car keypoint localization

Based on the architecture proposed in the ECCV 2016 paper [Stacked Hourglass Networks for Human Pose Estimation] (https://arxiv.org/abs/1603.06937)

Currently, the repository contains code for testing the network for car keypoint localization. Training code will be updated.

The code is written in torch, heavily borrows from https://github.com/anewell/pose-hg-train and has been modified to work on multi-GPU configurations.

To get started, run the prediction code in predictKpsLight.lua by executing
```
th predictKpsLight.lua
```

This script expects an input .txt file containing data in a format similar to that in `testInstances.txt`.
Specifically, each line contains one input instance in the following format
```
/abs/path/to/the/image.png x y w h
```
Here, `x y w h` are the `(x,y)` coordinates of the top left corner, width, and height respectively of the bounding box containing a car in the original image dimensions. The script takes care of the necessary cropping, scaling, transformations.
The script writes output (detected keypoints and confidence scores) to `results.txt`. These keypoints are according to the `64 x 64` cropped image size. These will have to be scaled to the original image size as per your requirements.
