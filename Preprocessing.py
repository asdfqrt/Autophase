import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image

df = pd.read_csv("data/For Training/metadata.CSV")
phaselist = df["primary_microconstituent"].unique()
dic = {phase: i for i,phase in enumerate(phaselist)}

class dataprepare():
    def __init__(self):
        self.df = pd.read_csv("data/For Training/metadata.CSV")
        self.img_dir = "data/For Training/train"

    # def delimage(self):
    #     dellist = []
    #     for file in os.listdir(self.img_dir):
    #         if (self.df["path"] == file).any():
    #             print(file,"O")
    #         else:
    #             dellist.append(file)

    #     for file in dellist:
    #         os.remove(self.img_dir+"/"+file)
    #         print(file,"삭제")
    def printinfo(self):
        groups = self.df.groupby('primary_microconstituent')['path'].apply(list).to_dict()
        for phase in groups:
            print("{}는 {}개 있으며 이는 전체 중 {:.2f}% 입니다".format(phase,len(groups[phase]),100*len(groups[phase])/len(self.df)))

    def augmentation(self,resize_level):
        # 최대 갯수의 class와 같은 갯수가 될때까지 trainsform된 copy 데이터 생성
        # 최대클래스 갯수/현재 클래스 갯수를 저장한 배열을 만든다  ex) [2, 1 ,1.4, 1.5, 3.1]
        # 만약 배열값이 2일 경우 현 클래스의 데이터에 대해 copy 데이터 1개씩 생성

        groups = {}
        for _, row in self.df.iterrows():
            file_path = os.path.join(self.img_dir, row['path'])
            if os.path.exists(file_path):
                microconstituent = row['primary_microconstituent']
                if microconstituent in groups:
                    groups[microconstituent].append(row['path'])
                else:
                    groups[microconstituent] = [row['path']]
        class_count = {phase: len(files) for phase, files in groups.items()}
        max_count =  max(class_count.values())
        traindata = {}

        trans = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]),
            transforms.Resize(resize_level),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for phase in class_count:
            ratio = max_count / class_count[phase]
            traindata[phase] = augmDataset(phase,trans,self.img_dir,ratio,groups)
        
        dataset = ConcatDataset(traindata.values())
        torch.save(dataset, 'dataaug.pt')


class augmDataset(Dataset):
    def __init__(self, phase, transform, img_dir, ratio, groups):
        self.phase = phase
        self.transform = transform
        self.img_dir = img_dir
        self.ratio = ratio
        self.groups = groups
        self.indices = list(range(len(groups[phase])))

    def __len__(self):
        return int(self.ratio * len(self.indices))

    def __getitem__(self, idx):
        idx = self.indices[idx % len(self.indices)]
        img_path = os.path.join(self.img_dir, self.groups[self.phase][idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, dic[self.phase]

def run(overwrite,resize_level):
    if overwrite:
        d = dataprepare()
        d.augmentation(resize_level)
    return torch.load('dataaug.pt')

