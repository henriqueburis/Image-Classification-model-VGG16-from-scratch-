'''VGG16 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class vgg16scratch(nn.Module):
    def __init__(self, num_classes=0):
        super(vgg16scratch, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,2), stride=(2,2)), # pool de janela quadrada de tamanho = 2, passo = 2
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    
    n_classe = 1

    net = vgg16scratch(n_classe)
    x = torch.randn(2,3,227,227)
    print(net(Variable(x)).size())

