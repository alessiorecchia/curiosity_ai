import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = T.device("cuda" if T.cuda.is_available() else "cpu")


class Encoder_mod(nn.Module):
    def __init__(self, in_channels):
        super(Encoder_mod, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
 
    def forward(self,x):
        x = T.from_numpy(x).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        # x = F.normalize(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x)) #size [1, 32, 3, 3] batch, channels, 3 x 3
        x = x.flatten(start_dim=1) #size N, 288
        return x
 
class Inverse_mod(nn.Module):
    def __init__(self, n_actions):
        super(Inverse_mod, self).__init__()
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,n_actions)
 
    def forward(self, state1,state2):
        state1= state1.to(device)
        state2 = state2.to(device)
        x = T.cat( (state1, state2) ,dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.softmax(x,dim=1)
        return x
 
class Forward_mod(nn.Module):
    def __init__(self, n_actions):
        super(Forward_mod, self).__init__()
        self.n_action = n_actions
        self.linear1 = nn.Linear(134,128)
        self.linear2 = nn.Linear(128,128)
 
    def forward(self,state,action):
        state = state.to(device)
        action = action.to(device)
        action_ = T.zeros(1,self.n_action)
        action_ = action_.to(device)
        indices = T.stack((T.arange(1, device=device), action), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = T.cat((state, action_) ,dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ICM():
    def __init__(self, encoder, forward_model, forward_criterion, inverse_model, inverse_criterion,
                forward_scale = 1., inverse_scale=1e4) -> None:
        self. encoder = encoder
        self.forward_model = forward_model
        self.forward_criterion = forward_criterion
        self.inverse_model = inverse_model
        self.inverse_criterion = inverse_criterion
        self.forward_scale = forward_scale
        self.inverse_scale = inverse_scale
    
    def predict(self,action, state1, state2):

        self.action = action
        self.state1 = state1
        self.state2 = state2

        state1_hat = self.encoder(self.state1)
        state2_hat = self.encoder(self.state2)
        state2_hat_pred = self.forward_model(state1_hat.detach(), self.action.detach())
        self.forward_loss = self.forward_criterion(state2_hat_pred,state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        forward_pred_err = self.forward_scale * self.forward_loss
        pred_action = self.inverse_model(state1_hat, state2_hat)
        self.inverse_loss = self.inverse_criterion(pred_action, self.action.detach().flatten()).unsqueeze(dim=1)
        inverse_pred_err = self.inverse_scale * self.inverse_loss
        return forward_pred_err, inverse_pred_err
    
'''self.
def ICM(state1, action, state2, forward_scale=1., inverse_scale=1e4):

    action = action.to(device)

    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
    forward_pred_err = forward_scale * forward_loss(state2_hat_pred,
                       state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat)
    inverse_pred_err = inverse_scale * inverse_loss(pred_action,\
                       action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err

def minibatch_train(use_extrinsic=True):
    # state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    # action_batch = action_batch.view(action_batch.shape[0],1)
    # reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
    forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch)
    i_reward = (1. / params['eta']) * forward_pred_err
    reward = i_reward.detach()
    if use_extrinsic:
        reward_batch = reward_batch.to(device)
        reward += reward_batch 
    qvals = model(state2_batch)
    reward += params['gamma'] * T.max(qvals)
    reward_pred = model(state1_batch)
    reward_target = reward_pred.clone()
    indices = T.stack( (T.arange(action_batch.shape[0]), action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss

def loss_fn(q_loss, inverse_loss, forward_loss):
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss

forward_pred_err, inverse_pred_err, q_loss = minibatch_train(use_extrinsic=use_explicit)
        loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err)
        loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
                        inverse_pred_err.flatten().mean())
        losses.append(loss_list)
        loss.backward()
        optimizer.step()

encoder = Encoder_mod()
encoder.to(device)
forward_model = Forward_mod()
forward_model.to(device)
inverse_model = Inverse_mod()
inverse_model.to(device)
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')

'''