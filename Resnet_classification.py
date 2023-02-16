import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

use_cuda = torch.cuda.is_available()
device = ("cuda" if use_cuda else "cpu")
torch.manual_seed(42)

######################
# File path
csv_path = "data/For Training/metadata.CSV"
train_img_dir = 'data/For Training/sample_train'
test_img_dir = 'data/For Training/sample_test'
######################
# Hyperparameters
batch_size = 16
nb_epochs = 10
learning_rate = 0.005
resize_level = (100,100)
model = models.resnet152(pretrained=True).to(device)
######################


df = pd.read_csv(csv_path)
phaselist = df["primary_microconstituent"].unique()
dic = {phase: i for i,phase in enumerate(phaselist)}
mapping = dict(zip(df['path'].tolist(),df['primary_microconstituent'].tolist()))
groups = df.groupby('primary_microconstituent')['path'].apply(list).to_dict()
class_count = {phase: len(files) for phase, files in groups.items()}
max_class = max(class_count, key=class_count.get)

class CustomDataset(Dataset):
    def __init__(self,img_dir, transform, augment=None, is_train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.is_train = is_train
        self.files = [f for f in os.listdir(img_dir) if f in df['path'].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        filename = self.files[idx]
        label = dic[mapping[filename]]
        # if self.is_train:
        #     if label != mode:
        #         image = self.augment(image)
        image = self.transform(image)
        return image, label

# class testDataset(Dataset):
#     def __init__(self,img_dir, transform):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.files = [f for f in os.listdir(img_dir) if f in df['path'].tolist()]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.files[idx])
#         image = Image.open(img_path).convert("RGB")
#         image = self.transform(image)
#         filename = self.files[idx]
#         label = dic[mapping[filename]]
#         return image, label

trans = transforms.Compose([transforms.Resize(resize_level),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
augment = transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                   ])
trainset = CustomDataset(img_dir = train_img_dir, is_train=True, transform=trans, augment=augment)

# trainset = ConcatDataset(trainlist)
testset = CustomDataset(img_dir = test_img_dir, is_train=False, transform=trans)



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
model.fc = nn.Linear(num_ftrs,len(dic)).to(device)

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
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x,y in loader_test:
            x=x.to(device)
            y=y.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            correct += predicted.eq(y.data.view_as(predicted)).sum()
            y_true += y.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()

    data_num = len(loader_test.dataset)
    print("테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)".format(correct,data_num,100 * correct/data_num))
    f1 = f1_score(y_true,y_pred,average='weighted')
    print("F1 score: {:.4f}".format(f1))
    
def present(limit):
    model.eval()
    with torch.no_grad():
        cnt = 1
        for x,y in loader_present:
            if cnt>limit: break
            cnt +=1
            x=x.to(device)
            y=y.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            image_array = (x.squeeze().permute(1,2,0).cpu().numpy() + 0.5).clip(0,1)
            plt.imshow(image_array)
            print("예측 결과 : {}, 실제 상태 : {}".format(phaselist[predicted],phaselist[y]))
            plt.show()
present(2)
test()
for epoch in range(nb_epochs):
    train(epoch)
test()
present(30)