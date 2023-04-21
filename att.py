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
from matplotlib.pyplot import show
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
test_dir = "./faces/testing/"
txt_root = "./attPath.txt" # 放所有att檔案的path
txt_root_test = "./attPathTest.txt"
train_batch_size = 10

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=45.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def convert(train=True):
    if(train):
        f=open(txt_root, 'w')
        data_path=training_dir
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i in range(37):
              for j in range(10):
                    img_path = data_path+'s'+str(i+3)+'/'+str(j+1)+'.pgm'
                    f.write(img_path+' '+str(i)+'\n')      
        f.close()

def convert_test(train, txt_root_test, test_dir):
    if(train!=True):
        f=open(txt_root_test, 'w')
        data_path=test_dir
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i in range(3):
              for j in range(10):
                    img_path = data_path+'s'+str(i+1)+'/'+str(j+1)+'.pgm'
                    f.write(img_path+' '+str(i)+'\n')      
        f.close()

class SiameseNetworkDataset(Dataset):
    def __init__(self, txt, dir, transform=None, target_transform=None, should_invert=False):  
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt       # 之前生成的train.txt
        self.dir = dir

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 随机选择一个人脸
        line.strip('\n')
        img0_list= line.split()
        should_get_same_class = random.randint(0,1)     # 随机数0或1，是否选择同一个人的脸，这里为了保证尽量使匹配和非匹配数据大致平衡（正负类样本相当）
        if should_get_same_class:    # 执行的话就挑一张同一个人的脸作为匹配样本对
                while True:
                    img1_list = linecache.getline(self.txt, random.randint(1, self.__len__())).strip('\n').split()
                    if img0_list[1]==img1_list[1]:
                            break
        else:       # else就是随意挑一个人的脸作为非匹配样本对，当然也可能抽到同一个人的脸，概率较小而已
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
    
        return img0, img1 , torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])],dtype=np.float32))    # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
    
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
    train_data = SiameseNetworkDataset(txt = txt_root, dir = training_dir,
                                    transform=transforms.Compose(
                [transforms.Resize((100,100)),transforms.ToTensor(),AddGaussianNoise(0., 45.)]
                ), 
                                    should_invert=False)     #Resize到100,100
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=2, batch_size = train_batch_size)



    train=False
    convert_test(train, txt_root_test, test_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork().to(device)
    test_data = SiameseNetworkDataset(txt = txt_root_test, dir = test_dir,
                                    transform=transforms.Compose(
                [transforms.Resize((100,100)),transforms.ToTensor(),AddGaussianNoise(0., 45.)]
                ), 
                                    should_invert=False)     #Resize到100,100

    test_dataloader = DataLoader(dataset=test_data, num_workers=6,batch_size=1,shuffle=True)


    #net = SiameseNetwork().cuda()     # GPU加速
    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)


    counter = []
    loss_history =[]
    iteration_number =0
    train_number_epochs = 14

    for epoch in range(0, train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)
            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i%10 == 0:
                print("Epoch:{},  Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    plt.plot(counter, loss_history)     # plot 损失函数变化曲线
    plt.show()

    accuracy=0
    counter=0
    correct=0
    for i, data in enumerate(test_dataloader,0): 
        x0, x1 , label = data
        # onehsot applies in the output of 128 dense vectors which is then converted to 2 dense vectors
        output1,output2 = model(x0.to(device),x1.to(device))
        res=torch.abs(output1 - output2)
        label=label[0].tolist()
        label=int(label[0])
        result=torch.max(res,1)[1].data[0].tolist()
        if label == result:
            correct=correct+1
        counter=counter+1
        if counter ==20:
            break
        
    accuracy=(correct/len(test_dataloader))*100
    print("Accuracy:{}%".format(accuracy))