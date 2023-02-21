🌍
*[English](README.md) ∙ [한국어](README-kr.md)*


# AutoPhase: Automated Steel Phase Classification using Pre-trained CNN Model
*Pytorch implementation of Phase classification of Steel using pre-trained CNN model.*

## Steel Phase Analysis
### For someone who are not familiar with material science

<img align="left" width="100" height="100" src="https://user-images.githubusercontent.com/79451613/220240779-df357d88-441b-48d2-bd2a-c095ba37e4c5.png">
Let's say you're designing a car. You want to make it lighter and stronger than the old ones, so you decide to use a new steel.

---

<img align="left" width="100" height="100" src="https://user-images.githubusercontent.com/79451613/220240788-002d12ee-1d7e-46da-aec7-903c26a3a5aa.png">
Combine different alloying elements and choose processing methods to create steel with the desired properties.
- like cooking: add salt for seasoning, or chili for heat.

---

<img align="left" width="100" height="100" src="https://user-images.githubusercontent.com/79451613/220240786-176eca1e-f84b-4240-a561-33ca81f1b850.png">
Now what? You want to make sure the steel you made is right. like tasting your food before serving it because it might be salty from too much salt or burnt from too much cooking time. That's what Phase Analysis do.

---

<img align="left" width="100" height="100" src="https://user-images.githubusercontent.com/79451613/220240783-ff50e301-c431-44f0-af05-00936cd26e8d.png">
Analyzing the microstructure of steel requires an expert to manually observe it using a microscope, which can be time-consuming and expensive.

---

<img align="left" width="100" height="100" src="https://user-images.githubusercontent.com/79451613/220240781-578af4d5-a18e-49ec-9c18-1e8b1e57d75a.png">
With Autophase, you don't need an expert - or even a human. Trained by deep learning, the tool can automatically determine the condition of steel very quickly and accurately using only images. 

- Industry:  Car, Construction, Electrical, Aerospace, Oil & Gas 

- Application: Product design, Quality control, Failure analysis, R&D

---

## Outline
![cnn str](https://user-images.githubusercontent.com/79451613/220237019-00eeae2a-ee84-435a-bfa3-0c2de3012ee8.png)

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
