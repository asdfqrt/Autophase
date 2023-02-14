# Fe phase classification

Pytorch implementation of Phase classification of high carbon steel using pre-trained CNN model.
This project is still in progress.

## Environmnet
- Python3
- Pytorch

## Getting stared
### Dataset
Place your Micrograph dataset in proper path

    Fe phase classification
    ├── data
    │    └── For Training
    │         ├── sample_train
    │         │    ├── xxx.png
    │         │    ├── yyy.png
    │         │    └── ...
    │         ├── sample_test
    │         │    ├── aaa.png
    │         │    ├── bbb.png
    │         │    └── ...
    │         └── metadata.csv
    └── Resnet_classification.py

metadata.csv must contain "path" and "primary_microconstituent" values of dataset

Or, you can just use sameple files in this repository for 

## Train
Run
    $ python
    
## Test
    $ python

## Result
Some fancy image descriptions
! The test result will be recored in /data/results.csv

## References
- Dataset: UHCSDB: UltraHigh Carbon Steel Micrograph DataBase(https://doi.org/10.1007/s40192-017-0097-0)
