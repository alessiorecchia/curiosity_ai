import torch as T
import torch.nn as nn
import torch.nn.functional as F

device = T.device("cuda" if T.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
 
class Inverse(nn.Module):
    def __init__(self):
        super(Inverse, self).__init__()
        self.linear1 = nn.Linear(576,256)
        self.linear2 = nn.Linear(256,12)
 
    def forward(self, state1,state2):
        x = T.cat( (state1, state2) ,dim=1)
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.softmax(x,dim=1)
        return x
 
class Forward(nn.Module):
    def __init__(self):
        super(Forward, self).__init__()
        self.linear1 = nn.Linear(300,256)
        self.linear2 = nn.Linear(256,288)
 
    def forward(self,state,action):
        state.to(device), action.to(device)
        action_ = T.zeros(action.shape[0],12)
        indices = T.stack((T.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = T.cat((state,action_) ,dim=1).to(device)
        # x = x.to(device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

