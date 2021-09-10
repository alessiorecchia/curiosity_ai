import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w))), kernel_size=3, stride=1)
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h))), kernel_size=3, stride=1)

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return self.head(x.view(x.size(0), -1))
    
    def conv2d_size_out(self, size, kernel_size = 5, stride = 2, padding = 1):
        return (size - kernel_size + 2 * padding) // stride  + 1