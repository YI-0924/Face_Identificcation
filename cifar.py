import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.autograd import Variable
import os

DOANLOAD_DATASET = True
LR = 0.001
BATCH_SIZE=128
EPOCH = 10
MODELS_PATH = './models'

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=45.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


train_transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, 4),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    AddGaussianNoise(0., 45.)
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    AddGaussianNoise(0., 45.)
])

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=True,
    transform=train_transform,
    download=DOANLOAD_DATASET
)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=False,
    transform=test_transform,
    download=DOANLOAD_DATASET
)

data_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CNN(nn.Module):
  def __init__(self, num_classes: int):
    super(CNN, self).__init__()
    self.num_classes = num_classes

    # in[N, 3, 32, 32] => out[N, 16, 16, 16]
    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        ),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2)
    )
    # in[N, 16, 16, 16] => out[N, 32, 8, 8]
    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, 5, 1, 2),
        nn.ReLU(True),
        nn.MaxPool2d(2)

    )
    # in[N, 32 * 8 * 8] => out[N, 128]
    self.fc1 = nn.Sequential(
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(True)
    )
    # in[N, 128] => out[N, 64]
    self.fc2 = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(True)
    )
    # in[N, 64] => out[N, 10]
    self.out = nn.Linear(64, self.num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
    x = self.fc1(x)
    x = self.fc2(x)
    output = self.out(x)
    return output
  
cnn = CNN(len(classes))

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
  cnn.train()
  for step, (x, y) in enumerate(data_loader):
    b_x = Variable(x, requires_grad=False)
    b_y = Variable(y, requires_grad=False)
    out = cnn(b_x)
    loss = loss_function(out, b_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
      print('Epoch: {} | Step: {} | Loss: {}'.format(epoch + 1, step, loss))

if not os.path.exists(MODELS_PATH):
  os.mkdir(MODELS_PATH)
torch.save(cnn, os.path.join(MODELS_PATH, 'cnn_model.pt'))

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=test_data.data.shape[0],
    shuffle=False
)
test_x, test_y = next(iter(test_loader))

cnn.eval()
prediction = torch.argmax(cnn(test_x), 1)
acc = torch.eq(prediction, test_y)
print('Accuracy: {:.2%}'.format((torch.sum(acc) / acc.shape[0]).item()))
