import torch as T
import numpy as np
import torch.multiprocessing as mp

from torch import nn
from torch.nn import functional as F
from torch.autograd  import Variable

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, n_actions):

        super(ActorCritic, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # print('self.input_size', self.input_size)
        
        # self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers=num_layers,
        #                     batch_first = True,  dropout = 0)
        
        self.rnn = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers=num_layers,
                            batch_first = True,  dropout = 0)

        self.fc1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.att = nn.Linear(self.hidden_size, n_actions)
        self.pre_val = nn.Linear(self.hidden_size, self.hidden_size)
        self.val = nn.Linear(self.hidden_size, 1)

    def forward(self, state):
        state = T.from_numpy(state).flatten().unsqueeze(0).unsqueeze(0).float().to(device)
        h0 = Variable(T.rand(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        # c0 = Variable(T.rand(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        # out, (hn, cn) = self.rnn(state.to(device), (h0.to(device), c0.to(device)))
        out, hn = self.rnn(state.to(device), h0.to(device))

        hidden_out = hn#[-1]

        x = F.relu(self.fc1(hidden_out))
        x = F.relu(self.fc2(x))

        scores = F.relu(self.att(x))
        scores = scores[-1].squeeze()

        probs = F.softmax(scores, dim=0)

        val = F.relu(self.pre_val(hidden_out.detach()))
        val = self.val(val[-1])

        index = T.multinomial(probs, 1)

        return index, scores, val


test = ActorCritic(3*60*60, 256, 6, 6)

test.to(device)

x = np.random.random(size=3*60*60)

index, scores, val = test(x)

print(index, scores, val)