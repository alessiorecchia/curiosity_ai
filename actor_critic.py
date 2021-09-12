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
        
        self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers=num_layers,
                            batch_first = True,  dropout = 0)

        self.fc1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.att = nn.Linear(self.hidden_size, n_actions)
        # self.act = nn.Linear(self.hidden_size, n_actions)
        self.val = nn.Linear(self.hidden_size, 1)

    def forward(self, state):

        state = T.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to(device)
        # h0 = Variable(T.zeros(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        # c0 = Variable(T.zeros(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        h0 = Variable(T.rand(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        c0 = Variable(T.rand(self.num_layers, state.shape[0], self.hidden_size), requires_grad = True)
        out, (hn, cn) = self.rnn(state.to(device), (h0.to(device), c0.to(device)))

        last_hidden = hn#[-1]

        hidden_out = self.relu(last_hidden)

        x = F.relu(self.fc1(hidden_out))
        x = F.relu(self.fc2(x))

        scores = F.relu(self.att(x))
        

        probs = F.softmax(scores, dim=0)

        val = self.val(hidden_out)

        index = T.multinomial(probs[-1], 1)

        # print(index)

        return index, scores, val


# test = ActorCritic(7500, 256, 6, 6)

# test.to(device)

# x = np.random.random(size=7500)

# index, scores, val = test(x)

# print(index, scores, val)