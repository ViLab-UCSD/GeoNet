import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self, pretrained=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*5*5, 500)
        # self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        if x.shape[1] == 1: ## handle MNIST
            x = torch.cat([x,x,x], dim=1)
            x = F.interpolate(x, size=(32,32))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        # x = F.adaptive_avg_pool2d(x , (1,1)).view(-1,50)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"

def lenet(**kwargs):
    '''
    return a lenet model CNN
    '''
    model = LeNet(**kwargs)
    return model