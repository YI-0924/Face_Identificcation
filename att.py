@@ -12,48 +12,42 @@ import os
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
        f=open(txt_root, 'w')
        data_path=training_dir
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i in range(40):
              for j in range(10):
                    img_path = data_path+'s'+str(i+1)+'/'+str(j+1)+'.pgm'
                    f.write(img_path+' '+str(i)+'\n')      
        f.close()

# 高斯模糊
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=Gaussian_factor):
    def __init__(self, mean=0., std=45.):
        self.std = std
        self.mean = mean
        
@ -64,24 +58,25 @@ class AddGaussianNoise(object):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SiameseNetworkDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, should_invert=False):  
    def __init__(self, txt, dir, transform=None, target_transform=None, should_invert=False):  
        self.transform = transform
        self.target_transform = target_transform
        self.should_invert = should_invert
        self.txt = txt       # 之前生成的train.txt
        self.dir = dir

    def __getitem__(self, index):
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 隨機選擇一個人臉
        line = linecache.getline(self.txt, random.randint(1, self.__len__()))   # 随机选择一个人脸
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
@ -96,7 +91,7 @@ class SiameseNetworkDataset(Dataset):
            img0 = self.transform(img0)
            img1 = self.transform(img1)
    
        return img0, img1 , torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])], dtype=np.float32))    # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
        return img0, img1 , torch.from_numpy(np.array([int(img1_list[1]!=img0_list[1])],dtype=np.float32))    # 注意一定要返回数据+标签， 这里返回一对图像+label（应由numpy转为tensor）
    
    def __len__(self):       # 数据总长
        fh = open(self.txt, 'r')
@ -125,7 +120,7 @@ class SiameseNetwork(nn.Module):
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
@ -135,7 +130,7 @@ class SiameseNetwork(nn.Module):
            nn.ReLU(inplace=True),
            
            nn.Linear(500, 5)
        )
            )

    def forward_once(self, x):
        output = self.cnn1(x)
@ -160,55 +155,32 @@ class ContrastiveLoss(torch.nn.Module):

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
    train_data = SiameseNetworkDataset(txt = txt_root,
                                       transform=transforms.Compose([
                                           transforms.Resize((100,100)),
                                           transforms.ToTensor(),
                                           AddGaussianNoise(0., 45.)
                                           ]), 
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
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, num_workers=2, batch_size = train_batch_size)

    #net = SiameseNetwork().cuda()     # GPU加速
    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)


    counter = []
    loss_history =[]
    iteration_number =0
    train_number_epochs = 10

    net.train()
    for epoch in range(1, train_number_epochs + 1):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)
            output1, output2 = net(img0, img1)

            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
@ -219,28 +191,31 @@ if __name__ == '__main__':
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
<<<<<<< HEAD
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
=======
    
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
    plt.show()     # plot 损失函数变化曲线
>>>>>>> parent of 1657ca6 (att: 整理function，並更改convert function)