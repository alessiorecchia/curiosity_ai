import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import instancenorm
from torch.autograd import Variable

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Base_Encoder(nn.Module):
    def __init__(self, in_dims):
        super(Base_Encoder, self).__init__()

        # encoder def
        self.channels = in_dims[0]
        self.img_dim = in_dims[1]
        self.hidden_size = 128
        in_size = self.get_input_size(self.get_input_size(self.get_input_size\
                         (self.get_input_size(self.img_dim, 5, 2, 2))))
        self.input_size = in_size**2 * 64
        
        # print('input size', self.input_size)

        self.enc_conv1 = nn.Conv2d(self.channels, 16, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.enc_rnn = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
    
    def get_input_size(self, in_dim, k_size=3, stride=2, pad=1):
        return ((in_dim - k_size + 2*pad)//stride + 1)
    
    def forward(self, obs):

        obs = T.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
        enc = F.elu(self.enc_conv1(obs))
        enc = F.elu(self.enc_conv2(enc))
        enc = F.elu(self.enc_conv3(enc))
        enc = F.elu(self.enc_conv4(enc))

        enc = enc.flatten().unsqueeze(0)
        h0 = Variable(T.rand(1, self.hidden_size), requires_grad = True)

        return self.enc_rnn(enc, h0.to(device))

class ICM(nn.Module):

    def __init__(self, in_dims, n_actions, **params) -> None:
        super(ICM, self).__init__()
        self.n_actions = n_actions
        try:
            self.fw_scale = params['forward_scale']
            self.inv_scale = params['inverse_scale']
        except:
            pass

        self.fw_criterion = nn.MSELoss(reduction='none')
        self.inv_criterion = nn.CrossEntropyLoss(reduction='none')

        self.encoder = Base_Encoder(in_dims)

        # forward model def
        self.fw_linear1 = nn.Linear(self.encoder.hidden_size + 1, 256)
        self.fw_output = nn.Linear(256, self.encoder.hidden_size)

        # inverse model def
        self.inv_linear1 = nn.Linear(2 * self.encoder.hidden_size, 256)
        self.inv_output = nn.Linear(256, n_actions)
    
    def forward(self, state, next_state, action):

        self.enc_state = self.encoder(state)
        self.enc_next_state = self.encoder(next_state)

        action = action.to(device)
        
        fw_in = T.cat((self.enc_state.squeeze(), action))
        enc_next_state_hat = F.relu(self.fw_linear1(fw_in))
        enc_next_state_hat = F.relu(self.fw_output(enc_next_state_hat))

        inv_in = T.cat((self.enc_state.squeeze(), self.enc_next_state.squeeze()))
        x = F.relu(self.inv_linear1(inv_in))
        x = F.relu(self.inv_output(x))
        self.inv_probs = F.softmax(x, dim=0)
        action_hat = T.multinomial(self.inv_probs, num_samples=1)
        
        return enc_next_state_hat, action_hat
    
    def predict(self,action, state1, state2):

        self.action = action
        self.next_state_hat, self.action_hat = self.forward(state1, state2, action)
        self.fw_loss = self.fw_criterion(self.next_state_hat, self.enc_next_state.squeeze().detach()).sum()
        forward_pred_err = self.fw_scale * self.fw_loss

        # oh_action = T.zeros(self.n_actions)
        # oh_action_hat = T.zeros(self.n_actions).to(device)

        # oh_action[action.item()] = 1
        # oh_action_hat[self.action_hat.item()] = 1

        # self.inv_loss = self.inv_criterion(oh_action_hat.unsqueeze(0), action.detach())#.flatten())#.unsqueeze(dim=1)
        self.inv_loss = self.inv_criterion(self.inv_probs.unsqueeze(0, ), action.detach())
        inverse_pred_err = self.inv_scale * self.inv_loss
        
        return forward_pred_err, inverse_pred_err

# x = np.random.random(size=(3, 42, 42))
# test = ICM(x.shape, 6)


# test.to(device)

# action = T.tensor([3])

# obs_hat, action_hat = test(x, action)

# print(obs_hat, action_hat)