import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = T.device("cuda" if T.cuda.is_available() else "cpu")


class Encoder_mod(nn.Module):
    def __init__(self):
        super(Encoder_mod, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
 
    def forward(self,x):
        x = x.to(device)
        x = F.normalize(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x)) #size [1, 32, 3, 3] batch, channels, 3 x 3
        x = x.flatten(start_dim=1) #size N, 288
        return x
 
class Inverse_mod(nn.Module):
    def __init__(self):
        super(Inverse_mod, self).__init__()
        self.linear1 = nn.Linear(576,256)
        self.linear2 = nn.Linear(256,12)
 
    def forward(self, state1,state2):
        state1= state1.to(device)
        state2 = state2.to(device)
        x = T.cat( (state1, state2) ,dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.softmax(x,dim=1)
        return x
 
class Forward_mod(nn.Module):
    def __init__(self):
        super(Forward_mod, self).__init__()
        self.linear1 = nn.Linear(300,256)
        self.linear2 = nn.Linear(256,288)
 
    def forward(self,state,action):
        state = state.to(device)
        action = action.to(device)
        action_ = T.zeros(action.shape[0],12)
        action_ = action_.to(device)
        indices = T.stack((T.arange(action.shape[0], device=device), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = T.cat((state,action_) ,dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

