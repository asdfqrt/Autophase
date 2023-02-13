import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = ("cuda" if use_cuda else "cpu")
torch.manual_seed(42)

#####################
batch_size = 16
nb_epochs = 10
learning_rate = 0.005
resize_level = (100,100)
model = models.resnet152(pretrained=True).to(device)
########################

trans = transforms.Compose([transforms.Resize(resize_level),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

df = pd.read_csv("data/For Training/new_metadata.CSV")
dic = {
    "spheroidite" : 0,
    "pearlite" : 1,
    "network" : 2,
    "spheroidite+widmanstatten" : 3,
    "pearlite+spheroidite" : 4,
    "pearlite+widmanstatten" : 5}
mapping = dict(zip(df['path'].tolist(),df['primary_microconstituent'].tolist()))

class CustomDataset(Dataset):
    def __init__(self,img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform

        self.files = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        filename = self.files[idx]
        label = dic[mapping[filename]]

        return image, label

trainset = CustomDataset(img_dir = 'data/For Training/sample_train', transform=trans)
testset = CustomDataset(img_dir = 'data/For Training/sample_test', transform=trans)

loader_train = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
loader_test = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
loader_present = DataLoader(
    dataset=testset,
    batch_size=1,
    shuffle=False
)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,6).to(device)

for name, param in model.named_parameters():
    if name not in ['fc.weight','fc.bias']:
        param.requires_grad = False

cost_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

total_batch = len(loader_train)



def train(epoch):
    model.train()
    avg_cost = 0
    for x,y in loader_train:
        x=x.to(device)
        y=y.to(device)
        prediction = model(x)
        cost = cost_fn(prediction,y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch
    
    print("Epoch:{} cost = {:.9f}".format(epoch+1,avg_cost))

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in loader_test:
            x=x.to(device)
            y=y.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            correct += predicted.eq(y.data.view_as(predicted)).sum()

    data_num = len(loader_test.dataset)
    print("테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)".format(correct,data_num,100 * correct/data_num))

def present(limit):
    phase=[
    "spheroidite",
    "pearlite",
    "network",
    "spheroidite+widmanstatten",
    "pearlite+spheroidite",
    "pearlite+widmanstatten"]
    model.eval()
    with torch.no_grad():
        cnt = 0
        for x,y in loader_present:
            if cnt>limit: break
            cnt +=1
            x=x.to(device)
            y=y.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            # plt.imshow(x.permute(100,100,3))
            print("예측 결과 : {}, 실제 상태 : {}".format(phase[predicted],phase[y]))
            # plt.show()
present(2)
test()
for epoch in range(nb_epochs):
    train(epoch)
test()
present(10)