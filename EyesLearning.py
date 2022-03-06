import torchvision
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import MobileNetV2
import torch.nn as nn
from tqdm import tqdm
from d2l import torch as d2l

# 读取数据并进行数据增强

# tips! 使用mobilenetv2进行finetune inputsize = 224
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.repeat(1,1,1)), # 由于图片是单通道的，所以重叠三张图像，获得一个三通道的数据
    #torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 读取数据 即读取每一张图片数据 并附上标签
def read_eyes_data(mode='train'):

    # 如果是训练集 需要读取 图片 labels train_augs
    if mode == 'train':
        dataset_csv = pd.read_csv('./Eys_Train_Dataset.csv',header=None)
        aug = train_augs
        label_arr = np.asarray(dataset_csv.iloc[1:,1])
    else:
        dataset_csv = pd.read_csv('./Eys_Test_Dataset.csv', header=None)
        aug = test_augs
    dataset_csv = dataset_csv.iloc[1:]
    #train_len = len(dataset_csv.index) - 1
    all_images,targets = [],[]
    for index,target in dataset_csv.iterrows():
        # img = target[1] labels = target[2]
        # 获取每张图片的像素值 这时候open不会直接加载到内存 但是进行list操作时就会加入到内存中 解决方法如下
        img_as_img = Image.open(target[1])
        #img_as_img = aug(img_as_img)
        # 这里open的是PIL Image图像 需要转换成 tensor 方法：np.array()
        all_images.append((img_as_img.copy()))
        # 转换成int型
        if mode == "train":
            label = int(target[2][0])
        else:
            label = 0
        targets.append(label)
        img_as_img.close()
    return all_images, targets


# 对每一个图片获取标签且进行图像增强
class EyesDataset(torch.utils.data.Dataset):
    # 需要继承torch.utils.data.Dataset类
    # 重载__init__  __getitem()__ __len__三个方法
    def __init__(self,mode='train',transform=None):
        self.mode = mode
        self.features,self.labels = read_eyes_data(mode)
        self.len = len(self.features)
        self.transform = train_augs if self.mode == 'train' else test_augs
        print('read ' + str(len(self.features)) + (f' training examples' if mode == 'train' else f' test examples'))
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        index = index % self.len
        if self.mode == 'test':
            image = self.features[index]
            image = self.transform(image)
            return image
        else:
            label = self.labels[index]
            image = self.features[index]
            # 对每张图片进行增强
            image = self.transform(image)
            #image = np.array(image)
            return (image,label)
    def __len__(self):
        return len(self.features)

# 数据加载器 ： 用来把训练数据局分成多个小组，每次抛出一组数据，直到把所有的数据都抛出。DataLoader 设置batch_size=64
batch_size = 128
def load_eyes_data(batch_size):
    train_iter = torch.utils.data.DataLoader(EyesDataset(mode='train'),batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(EyesDataset(mode='test'),batch_size)
    return train_iter,test_iter


# --------------------- train & test ---------------------

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# read data_iter
train_iter,test_iter = load_eyes_data(batch_size)

def get_device():
    return 'cuda' if torch.cuda.is_available() else "cpu"

def accuracy(y_hat,y):
    #y_hat = np.argmax(y_hat,axis=1)
    #cmp = y_hat.astype(y.dtype) == y
    cmp = torch.eq(y_hat,y).sum().float().item()
    return cmp

# use gpu to train
device = get_device()
# define model
model = MobileNetV2.mobilenet_v2(pretrained=True)
# Fatigue Detection中只有open eyes and closd eyes 2个output
model.classifier = nn.Linear(1280,2)
model.to(device)
# Optimizer and criterion 优化器 & loss
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
epochs = 10
best_acc = 0.0
train_steps = len(train_iter)


def train_batch(net,X,y,loss,trainer,devices):
    if isinstance(X,list):
        X = [x.to(devices) for x in X]
    else:
        X = X.to(devices)
    if isinstance(y,list):
        y = [Y.to(devices) for Y in y]
    else:
        y= y.to(devices)
    y = y.to(devices)
    net.train()
    # 清空梯度
    trainer.zero_grad()
    # 预测结果
    logits = net(X)
    l = loss(logits,y)
    # 反向传播
    l.sum().backward()
    # 优化器优化参数
    trainer.step()
    train_loss_sum = l.sum()
    pred = logits.argmax(dim=1)
    train_acc_sum = accuracy(pred,y)
    return train_loss_sum,train_acc_sum

def train_epoches(net):
    net.to(get_device())
    for epoch in range(epochs):
        train_loss = []
        train_accs = []
        metric = Accumulator(4)
        running_loss = 0.0
        train_bar = tqdm(train_iter)
        for step,data in enumerate(train_bar):
            images,labels = data
            l,acc = train_batch(net,images,labels,loss_f,optimizer,device)
            metric.add(l,acc,labels.shape[0],labels.numel())
            train_accs.append(acc)
        train_loss = metric[0] / len(train_accs)
        train_acc = metric[1] / metric[3]
        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    print("Finished Training")

def test_and_pred(net):
    net.eval()
    predictions = []
    for i,X in enumerate(test_iter):
        with torch.no_grad():
            pred = net(X.to(device))
        predictions.extend(pred.argmax(dim=-1).cpu().numpy().tolist())
    preds = []
    for i in predictions:
        preds.append(i)
    test_data = pd.read_csv('Eys_Test_Dataset.csv')
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['data_url'],test_data['label']],axis=1)
    submission.to_csv('submission.csv',index=False)


# --------------------- show & save result ---------------------


if __name__ == "__main__":
    #train_epoches(model)
    #test_and_pred(model)
    # 随便展示一张图片
    onebatch = next(iter(test_iter))
    read_result = pd.read_csv("submission.csv")
    titles = read_result['label'][0:10]
    titles = list(titles)
    imgs = (onebatch.permute(0,2,3,1))
    axes = d2l.show_images(imgs,2,5,titles=titles,scale=2)
    d2l.plt.show()
    #read_eyes_data()