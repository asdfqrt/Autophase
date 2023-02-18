import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import Preprocessing

use_cuda = torch.cuda.is_available()
device = ("cuda" if use_cuda else "cpu")
torch.manual_seed(42)

######################
# File path
csv_path = "data/For Training/metadata.CSV"
train_img_dir = 'data/For Training/train'
val_img_dir = 'data/For Training/val'
test_img_dir = 'data/For Training/test'
Imbalance_correction, Overwrite = True, True
######################
# Hyperparameters
batch_size = 16
nb_epochs = 10
learning_rate = 0.005
resize_level = (200,200)
model = models.resnet152(pretrained=True).to(device)
######################

print("Loading dataset...")
df = pd.read_csv(csv_path)
phaselist = df["primary_microconstituent"].unique()
dic = {phase: i for i,phase in enumerate(phaselist)}
mapping = dict(zip(df['path'].tolist(),df['primary_microconstituent'].tolist()))

class CustomDataset(Dataset):
    def __init__(self,img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.files = [f for f in os.listdir(img_dir) if f in df['path'].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        filename = self.files[idx]
        label = dic[mapping[filename]]
        return image, label

class testDataset(Dataset):
    def __init__(self,img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.files = [f for f in os.listdir(img_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.files[idx]


trans = transforms.Compose([transforms.Resize(resize_level),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])                            
if Imbalance_correction:
    trainset = Preprocessing.run(Overwrite,resize_level)
else:
    trainset = CustomDataset(img_dir = train_img_dir,transform=trans)
valset = CustomDataset(img_dir = val_img_dir, transform=trans)
testset = testDataset(img_dir=test_img_dir, transform=trans)


loader_train = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
loader_val = DataLoader(
    dataset=valset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
loader_test = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
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

def val():
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x,y in loader_val:
            x=x.to(device)
            y=y.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            correct += predicted.eq(y.data.view_as(predicted)).sum()
            y_true += y.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()

    data_num = len(loader_val.dataset)
    print("Validation 데이터에서 예측 정확도: {}/{} ({:.0f}%)".format(correct,data_num,100 * correct/data_num))
    f1 = f1_score(y_true,y_pred,average='weighted')
    print("F1 score: {:.4f}".format(f1))

def test():
    model.eval()
    results = []
    with torch.no_grad():
        for x,filename in loader_test:
            x=x.to(device)
            prediction = model(x)
            _,predicted = torch.max(prediction.data,1)
            for file, pred in zip(filename, predicted):
                results.append((file, phaselist[pred.item()]))

    results_df = pd.DataFrame(results, columns=['Image file name','predict'])
    results_df.to_csv('data/results.csv', index=False)
                

def present(limit):
    model.eval()
    with torch.no_grad():
        cnt = 0
        num_rows = 2
        num_cols = 5
        while cnt < limit:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 4))
            for row in range(num_rows):
                for col in range(num_cols):
                    if cnt >= limit:
                        break
                    data = next(iter(loader_val))
                    x = data[0].to(device)
                    y = data[1].to(device)
                    prediction = model(x)
                    _, predicted = torch.max(prediction.data, 1)
                    image_array = (data[0][0].permute(1, 2, 0).cpu().numpy() + 0.5).clip(0, 1)
                    axs[row][col].imshow(image_array)
                    axs[row][col].set_title("Predict : {}\n Real : {}".format(phaselist[predicted[0]], phaselist[y[0]]))
                    axs[row][col].axis("off")
                    cnt += 1
            for i in range(cnt, num_cols*num_rows):
                row = i // num_cols
                col = i % num_cols
                axs[row][col].axis("off")
            plt.subplots_adjust(wspace=1.0, hspace=0.5)
            plt.tight_layout()
            plt.show()

######################################
Preprocessing.dataprepare().printinfo()
print("Before training")
val()
print("Start training")
for epoch in range(nb_epochs):
    train(epoch)
print("Training Finish")
val()
test()
present(100)