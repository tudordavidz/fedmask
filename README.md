# Raspberry Pi 4 Computer Vision and Federated Learning Framework

This repository contains a Computer Vision and Federated Learning framework designed to run optimally on a Raspberry Pi 4 (8 GB RAM) with a 64-bit Bullseye operating system.

## Prerequisites

1. **Raspberry Pi Setup**

   - Ensure you have a Raspberry Pi 4 with 8 GB RAM running the **64-bit Bullseye** operating system.

2. **TensorFlow Installation**

   - Install **TensorFlow 2.14** by following the tutorial [here](https://qengineering.eu/install-tensorflow-on-raspberry-64-os.html).
   - To avoid installation errors, it is highly recommended to use an image for the Raspberry Pi that already has TensorFlow and OpenCV pre-installed. You can find this image [here](https://github.com/Qengineering/RPi-Bullseye-DNN-image).

3. **Python Version**
   - Ensure you have **Python 3.9** installed on your Raspberry Pi.

## Installation

After setting up your Raspberry Pi, install the required Python packages using the following commands:

```bash
pip3 install opencv-python
pip3 install imutils
pip3 install scikit-learn
pip3 install python-dotenv
pip3 install matplotlib
```

## Computer Vision Framework

To run the Computer Vision framework, simply execute the following command:

```bash
python3 Start_System.py
```

This will create a new folder named **dataCollection** in the root of the project, containing data collected during the execution of the mask detection framework.

The Computer Vision framework utilizes two models: <br>

- FLModel.h5: A model trained using the Federated Learning framework.
- caffeFaceModel: A face detection model based on OpenCV’s Caffe framework. You can find more details [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).

## Federated Learning Framework

First step is to add the dataset from Kaggle. Please add randomly 500 images for each folder (with_mask, without_mask) and delete the .gitkeep file. The dataset used is [here](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset?resource=download).

To configure the Federated Learning framework, set the following global parameters in the .env file:

- NR_CLIENTS: Number of simulated clients
- COMMS_ROUND: Number of federated learning communication rounds
- NR_SPLITS: Number of splits for repeated K-folding
- NR_REPEATS: Number of repeats for repeated K-folding
- EPOCHS: Number of epochs for the centralized network
- DATASET_PATH: Path to the dataset used for both the Federated Learning and centralized network

To start the Federated Learning framework, run: <br>

```bash
python3 start.py
```

To start the centralized network framework, run: <br>

```bash
python3 centralized.py
```

## Results

Upon completion of the training, the following plots will be generated:

For Federated Learning Framework: <br>

- Model performance metrics
- Box and whisker plot
- Confusion matrix plot
- A table containing the mean with standard deviation, confidence intervals, and p-values.

Centralized Network Framework: <br>

- Model performance plot

All plots are saved here: ./Federated Learning/framework
