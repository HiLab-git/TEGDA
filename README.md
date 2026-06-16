# TEGDA
---
This is the official code for MICCAI 2025 paper "TEGDA: Test-time Evaluation-Guided Dynamic Adaptation for Medical Image Segmentation".

## To do
- [x] TEGDA code is updated

## Overall Framework
![](pictures/pipeline.png)

Our contributions are summarized as follows:
- We present a novel prediction quality evaluation metric based on **Agreement with Dropout Inferences calibrated by Confidence (ADIC)**, where the Dice score between predictions by the model and its dropout version is leveraged to assess the robustness of the model on a testing sample, then it is further calibrated by the confidence to become highly relevant to the real Dice value between the prediction and its ground-truth
- We propose **Adaptive Feature Fusion-based Refinement (AFFR)** that adaptively fuses the feature of a sample with those with high ADIC values based on their similarity, leading to robust refined pseudo-labels.
- We introduce ADIC-guided **Self-adaptive Model Updating (SMU)** that consists of ADIC-aware pseudo-label loss weighting and ADIC-aware mean teacher to improve the stability of adaptation.

## Dataset
Download the BraTS-GLI and BraTS-PED datasets from [BraTS 2023](https://www.synapse.org/#!Synapse:syn51156910/wiki/), M&Ms datasets from [M&Ms](http://www.ub.edu/mnms). We also provided the preprocessed version for TEGDA at Google Drive: [BraTS 2023](https://drive.google.com/drive/folders/1PNGLAzZg336s7JrN1-Up4QTLLzOR0-6C?usp=sharing) and [M&Ms](https://drive.google.com/file/d/10qWKjURSEp1Acx7RZsBEmAb3gr9mNNPR/view?usp=sharing), please download and extract the .zip file to the corresponding folder and generate the .csv file according to your root dir.

## How to use
### Create the visual enviroment
Use
```
conda env create -f environment.yaml
conda activate TTA
```
to setup the visual environment for the code.
### Source model training
Use
```
cd code
python train_fully_supervised_2D.py # For M&Ms dataset
python train_fully_supervised_3D.py # For BraTS dataset
```
to get the source model for two datasets.

### Test-time adaptation
Use
```
./run.sh
```
to get the test-time adaptation results on two datasets.
