# GBOT: Graph-Based 3D Object Tracking for Augmented Reality-Assisted Assembly Guidance

## Publication in proceding
[Arxiv](https://arxiv.org/pdf/2402.07677.pdf)


## Introduction
Guidance for assemblable parts is a promising field for the use of augmented reality. Augmented reality assembly guidance requires 6D object poses of target objects in real-time. Especially in time-critical medical or industrial settings, continuous and markerless tracking of individual parts is essential to visualize instructions superimposed on or next to the target object parts. In this regard, occlusions by the user's hand or other objects as well as the complexity of different assembly states complicate robust and real-time markerless multi-object tracking. 

![title](asset/teaser_proposal.png)

## Synthetic Generation
The first step is to install [BlenderProc](https://github.com/DLR-RM/BlenderProc).
To install the local project:

    cd BlenderProc
    pip install -e .

Some additional packages are required but since BlendeProc runs in its own environment they have to be installed there.
Use:

    blenderproc pip install tqdm
    blenderproc pip install rasterio

Next step is to download the newest textures for the backgrounds:

    python blenderproc download cc_textures resources/cctextures    

Everything else what is needed is saved in the "Synthetic_data_generation/resources" folder.
Now the synthetic data generation should be ready to run. See therefore the next two sections.

Also if you want to debug see the "BlenderProc/README.md" file.

## GBOT dataset
Comming soon

## Object Detection and 6D Pose Estimation
**1.** Install pytorch from https://pytorch.org/ by using the command provided there. Installation of pytorch before Ultralytics is important!

**2.** Install YOLOv8 by using:

    pip install ultralytics

For further instructions see: https://github.com/ultralytics/ultralytics

Additional packages will probably also be necessary but just install them based on their error message.

**3.** Probably you have to change the yolo settings. Use in the command line:
    
    yolo settings

to get the path where the settings are located and then change the "datasets_dir", "weights_dir" and "runs_dir" to your needs.
Recommended where PATH_TO_PROJECT is the Path where the project is located on your machine:

    datasets_dir: PATH_TO_GBOTdatasets
    weights_dir: PATH_TO_PROJECT\yolov8\weights  
    runs_dir: PATH_TO_PROJECT\yolov8\runs 

**4.** Start training with the script (You can skip this step if you want to use our pretrained models):

 python yolov8/yolov8_pose_training.py

 Export the model into onnx format:

  from ultralytics import YOLO
  # Load a model
  model = YOLO('yolov8pose.pt')  # load an official model
  model = YOLO('path/to/best.pt')  # load a custom trained model

  # Export the model
  model.export(format='onnx')

**5.** Predict 6D object pose with YOLOv8

Download our [pretrained models](https://zenodo.org/records/10688659) in onnx format. 
save the pretrained models in the folder: yolov8pose/pretrained

 python yolov8pose/yolo_to_pose_prediction_cv2.py


## 3D Object Tracking
