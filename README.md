🌍
*[English](README.md) ∙ [한국어](README-kr.md)*


# AutoPhase: Automated Steel Phase Classification using Pre-trained CNN Model
*Pytorch implementation of Phase classification of Steel using pre-trained CNN model.*

![infogra1](https://user-images.githubusercontent.com/79451613/220185276-3c6c9ba2-b594-4b9a-9785-90dd60598cae.png)
![outline](https://user-images.githubusercontent.com/79451613/219885029-596707b1-806a-4fc2-85c7-c6eea6dbc51e.png)

Autophase is an automated classification system for identifying Steel phases in Scanning Electron Microscope (SEM) images using deep learning. This project aims to overcome the difficulties in the traditional manual classification method, which is both time-consuming and error-prone. By leveraging a pre-trained convolutional neural network (CNN) model and fine-tuning it on the dataset, the system can achieve high classification accuracy, **91%**.

## Environmnet
- Python3
- Pytorch

## Getting Started
### Dataset
Place your Micrograph dataset in proper path

    Fe phase classification
    ├── data
    │    └── For Training
    │         ├── train
    │         │    ├── xxx.png
    │         │    ├── yyy.png
    │         │    └── ...
    │         ├── val
    │         │    ├── aaa.png
    │         │    ├── bbb.png
    │         │    └── ...
    │         ├── test
    │         │    ├── 111.png
    │         │    ├── 222.png
    │         │    └── ...
    │         └── metadata.csv
    └── Resnet_classification.py

metadata.csv must contain "path" and "primary_microconstituent" values of dataset

Or, you can just use sameple files in this repository for quick testing

### Hyperparameters
In `Resnet_classification.py`, you can change
* File path
* Imbalance_correction, Overwrite
* batch_size
* learning_rate
* resize_level
* epochs
* model

It's okay to use the default

## Data Augmentation
If your dataset is biased, model's predictive performance may suffer.

To correct the imbalance, this program has a data augmentation feature.

This is a feature that transforms and copies images from classes with a small amount of images and trains it as much as a major class.
The Data Augmentation data will be saved as `dataaug.pt`.

* If your training dataset is already balanced, to off the data augmentation 
`Imbalance_correction, Overwrite = False,False`

* If this is not your first time using this program and the dataaug.pt file already exists
`Imbalance_correction, Overwrite = True,False`

## Run
Run `Resnet_classification.py`
It will automatecally train & test images in the directory folders.

1. Print out the distribution of your training set
2. Accuracy before training
3. Train images
4. Accuracy after training
5. Visualization(default set to 100 images)

* The classification result will be recored in data/results.csv

## Result
![image (2)](https://user-images.githubusercontent.com/79451613/219881948-f062f3ab-4b01-42e8-a794-cd4cc251b267.png)

* The classification accuracy of this project was **91%**. This high level of accuracy demonstrates the effectiveness of my approach and its potential to be applied in real-world research and industrial settings.
* Expect higher accuracy depending on your dataset (higher resolution, dataset size..) and the model(ex.Coca) you use.
## References
- Dataset: [UHCSDB: UltraHigh Carbon Steel Micrograph DataBase](https://www.kaggle.com/datasets/safi842/highcarbon-micrographs)

## Autor
[asdfqrt](https://github.com/asdfqrt) / forsecretactive@gmail.com
