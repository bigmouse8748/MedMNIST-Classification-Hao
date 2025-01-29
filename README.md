# MedMNIST Classification Pipeline

An end-to-end pipeline for classifying medical images from the MedMNIST dataset. This repository demonstrates how to preprocess medical imaging data, train a deep learning model, and evaluate its performance.

---

## Table of Contents
- [About](#about)
- [Setup Instructions](#setup-instructions)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## About

This project processes images from the MedMNIST dataset to classify medical imaging data. It includes:
- Data preprocessing and augmentation.
- A ResNet-based deep learning model for classification.
- Performance evaluation with metrics and visualizations.

---
### Prerequisites
- Python 3.8+
- matplotlib==3.10.0
- medmnist==3.0.2
- numpy==2.2.2
- pandas==2.2.3
- scikit_learn==1.6.1
- torch==2.5.1+cu124
- torchvision==0.20.1+cu124
- tqdm==4.67.1
---
### Code Structure
* ['data/'](data/):
    * ['preprocess.py'](data/preprocess.py): Load selected dataset and perfom preprocessing for training, validation, and test.
    * ['preview.py'](data/preview.py): Preview selected dataset: Randomly select 100 images from all classes, or select specific number of images for each class.

* ['models/'](models/):
    * ['inception_resnet.py'](models/inception_resnet.py): Load InceptionResNet model built from scratch
    * ['stock_model.py'](models/stock_model.py): Load pretrained model provided by Pytorch
    * ['train.py'](models/train.py): Perform training with several selectable arguments.

* ['outputs/'](outputs/): Save training outputs in sub-folders. The   name of subfolder is created by the format of {dataset}_{model_name}_{created_time}. Training configurations, training logs, and model weights are stored in this folder.

* ['tests/'](tests/):
    * [test_results/](tests/test_results/): Saving the test result of each test event. Subfolder name with the format of {dataset}_{model_name}_{Datetime} saves the test result for "single model, single dataset" test pattern. Subfolder "Group_results" saves the test results for "multi model, single dataset" test pattern. 
    * ['test.py'](tests/test.py): Perform test with several selectable arguments.
* ['utils/'](utils/utils.py): helper functions.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bigmouse8748/MedMNIST-Classification-Hao.git
2. cd MedMNIST-Classification-Hao
### Usage
#### 1. Preview the data
 - cd data
 - python preview.py -h

