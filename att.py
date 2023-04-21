import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch import optim
import random
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import linecache
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import time
import copy
from torch.optim import lr_scheduler
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn.functional as F

training_dir = "./faces/training/"
testing_dir = "./faces/testing/"
training_txt_root = "./attPath-Train.txt" # 放所有att檔案的path
testing_txt_root  = "./attPath-Test.txt" # 放所有att檔案的path
test_dir = "./faces/testing/"
txt_root = "./attPath.txt" # 放所有att檔案的path
train_batch_size = 10
train_number_epochs = 10
MODELS_PATH = './models'
LR = 0.0005
Gaussian_factor = 15

# 把所有att檔案的path 寫到同個txt裡
def convert(train=True):
    if(train):
        txt_root = training_txt_root
        data_path = training_dir
    else:
        txt_root = testing_txt_root
        data_path = testing_dir

    if(not os.path.exists(data_path)):
        os.makedirs(data_path)
    
    f=open(txt_root, 'w')
    # 參考 https://newaurora.pixnet.net/blog/post/228299675-python%E8%AE%80%E5%8F%96%E8%B3%87%E6%96%99%E5%A4%BE%E5%85%A7%E6%AA%94%E6%A1%88
    # walk 會將指定路徑底下所有的目錄與檔案都列出來(包含子目錄以及子目錄底下的檔案)
    allList = os.walk(data_path)

    num = 0
    # 列出所有子目錄與子目錄底下所有的檔案
    for root, dirs, files in allList:
        # dir = root.split('/')[-1][1:]
        for i in files:
            if(i[0] == 'R'): continue # README不要讀進去
            img_path = root + '/' + i
            f.write(img_path + ' ' + str(num) +'\n')  
            num += 1    
    f.close()

# 高斯模糊
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=Gaussian_factor):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SiameseNetworkDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):  
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt       # 之前生成的train.txt

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 隨機選擇一個人臉
        line.strip('\n')
        img0_list= line.split()
        should_get_same_class = random.randint(0,1)     # 隨機數0或1，是否選擇同一人的臉，這裡為了導正盡量使匹配和非匹配數據大致平衡 (正負類樣本相當) 
        if should_get_same_class:   # 執行的話就挑一張同一個人的臉作為匹配樣本對
            while True:
                img1_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
                if img0_list[1]==img1_list[1]:
                    break
        else:       # else就是随意挑一個人的臉作為非匹配樣本對，當然也可能抽到同一個人的臉，概率較小而已
            img1_list = linecache.getline(self.txt, random.randint(1,self.__len__())).strip('\n').split()
        
        img0 = Image.open(img0_list[0])    # img_list都是大小为2的列表，list[0]为图像, list[1]为label
        img1 = Image.open(img1_list[0])
        img0 = img0.convert("L")           # 转为灰度
        img1 = img1.convert("L")
    
        if self.should_invert:             # 是否进行像素反转操作，即0变1,1变0
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:     # 非常方便的transform操作，在实例化时可以进行任意定制
            img0 = self.transform(img0)
            img1 = self.transform(img1)
    
        return img0, img1 , torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])], dtype=np.float32))    # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
    
    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
                
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            
            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


if __name__ == '__main__':
    if input("write txt? [press enter with NO rewrite]: "): # 寫檔
        convert(train=True)
        convert(train=False)
    
    print("Total epoch:", train_number_epochs,",  Batch size:",train_batch_size, ",  LR:",LR, ",  Gaussian Factor:", Gaussian_factor)
    test_transform=transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        # torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        AddGaussianNoise(0., 45.)
    ])
    
    train_data = SiameseNetworkDataset(txt = training_txt_root,
                                       # train=True,
                                       transform=test_transform, 
                                       should_invert=False)     #Resize到100,100
    
    test_data = SiameseNetworkDataset(txt = testing_txt_root,
                                       # train=False,
                                       transform=test_transform, 
                                       should_invert=False)     #Resize到100,100
    
    
    train_dataloader = DataLoader(
        dataset=train_data, 
        shuffle=True, 
        num_workers=2, 
        batch_size = train_batch_size
    )

    #net = SiameseNetwork().cuda()     # GPU加速
    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)


    counter = []
    loss_history =[]
    iteration_number =0

    net.train()
    for epoch in range(1, train_number_epochs + 1):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)
            output1, output2 = net(img0, img1)

            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i%10 == 0:
                print("Epoch:{},  Current loss {}".format(epoch,loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    torch.save(net, os.path.join(MODELS_PATH, 'att_model.pt'))
    
    """
    test_loader = DataLoader(
        dataset=test_data, 
        shuffle=False, 
        num_workers=2, 
        batch_size = len(test_data) # train_batch_size
    )

    test_x, test_y = next(iter(test_loader))

    net.eval()
    prediction = torch.argmax(net(test_x), 1)
    acc = torch.eq(prediction, test_y)
    print('Accuracy: {:.2%}'.format(
            (torch.sum(acc) / acc.shape[0]).item())
         )
    """

    plt.plot(counter, loss_history)
    plt.show()     # plot 損失函數變化曲線