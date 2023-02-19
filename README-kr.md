🌍
*[English](README.md) ∙ [한국어](README-kr.md)*


# AutoPhase: Pre-trained CNN 모델을 이용한 자동화 된 Fe의 상 분류
![outline](https://user-images.githubusercontent.com/79451613/219885029-596707b1-806a-4fc2-85c7-c6eea6dbc51e.png)

사전 학습된 CNN 모델을 사용하여 철의 상(相) 분류를 Pytorch로 구현하였습니다.

철의 기계적, 화학적 특성에 대한 수많은 연구에도 불구하고 철의 상 분류에는 여전히 전문가가 미세구조를 분석하는 수작업이 필요하며, 이는 시간과 비용이 많이 소요됩니다. 이 프로젝트는 딥러닝의 이미지 분석 기능을 활용하여 SEM 이미지에서 철의 상을 자동으로 분류하는 것에 대한 내용입니다.

## 환경
- Python3
- Pytorch

## 시작하기
### 데이터셋
학습 및 분류할 미세구조 사진과 메타 데이터를 올바른 경로에 배치하세요

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

metadata.csv는 사진의 파일명 "path"와 상 "primary_microconstituent"의 값이 포함되어야 합니다

* 다른 설정없이 이 저장소에 있는 파일들을 그대로 다운받기만 해도 테스트 해볼 수 있습니다

### 하이퍼 파라미터
`Resnet_classification.py` 에서 해당 값들을 조절할 수 있습니다.
* File path
* Imbalance_correction, Overwrite
* batch_size
* learning_rate
* resize_level
* epochs
* model

기본 값으로 사용해도 괜찮습니다.

## 데이터 증강
편향된 데이터 세트를 사용하는 경우 모델의 예측 성능이 저하될 수 있습니다.(상에 따라 이미지의 갯수가 다른 경우를 의미합니다)

이러한 불균형을 바로잡기 위해 이 프로그램에는 데이터 증강 기능(Data Augmentation)이 있습니다.

마이너 클래스의 이미지를 변환 및 복사하여 주요 클래스만큼 학습시키는 기능입니다.
증강된 데이터는 `dataaug.pt`라는 이름으로 저장됩니다.

* 만약 사용하는 데이터가 이미 균형 잡혀있어, 데이터 증강을 사용하고 싶지않다면
`Imbalance_correction, Overwrite = False,False`

* 이미 데이터 증강파일 `dataaug.pt`가 생성되었고, 추가적인 연산없이 빠르게 학습(ex.하이퍼파라미터 튜닝등을 위해)을 하고싶다면
`Imbalance_correction, Overwrite = True,False`

## 실행
`Resnet_classification.py` 를 실행하세요
자동으로 폴더 내의 이미지들을 학습하고 예측을 진행합니다.
프로그램은 아래와 같은 순서로 동작합니다

1. 학습데이터의 분포 출력
2. 학습 전 정확도 출력
3. 이미지 학습
4. 학습 후 정확도 출력
5. 결과 시각화(기본값으로 100장의 이미지를 보여줍니다)

* Test 데이터의 분류 결과는 data/results.csv에 저장됩니다

## 결과
![image (2)](https://user-images.githubusercontent.com/79451613/219881948-f062f3ab-4b01-42e8-a794-cd4cc251b267.png)

* 이 프로젝트는 **91%**의 분류 정확도를 달성하여 실 연구 및 산업현장에 적용될 수 있을만큼 효율적입니다.
* 데이터 셋의 해상도나 크기 개선, 타 모델(ex.CoCa) 적용을 통해 정확도를 개선 할 수 있습니다.

## 참조
- 사용된 Fe 이미지 데이터: [UHCSDB: UltraHigh Carbon Steel Micrograph DataBase](https://www.kaggle.com/datasets/safi842/highcarbon-micrographs)

## 제작자
[asdfqrt](https://github.com/asdfqrt)