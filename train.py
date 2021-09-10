import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize
from collections import deque


from curiosity import Encoder, Forward, Inverse
from memory import MemoryReplay
from dqn import DQN
from gameplay import GameField

device = T.device("cuda" if T.cuda.is_available() else "cpu")


params = {
    'batch_size':128,
    'beta':0.2,
    'lambda':0.1,
    'eta': 1.0,
    'gamma':0.2,
    'max_episode_len':3000,
    # 'min_progress':15,
    'action_repeats':6,
    'frames_per_state':3
}

env = GameField()
n_actions = env.action_space.n
replay = MemoryReplay(N=1000, batch_size=params['batch_size'])
model = DQN(42, 42, n_actions)
model.to(device)
encoder = Encoder()
encoder.to(device)
forward_model = Forward()
forward_model.to(device)
inverse_model = Inverse()
inverse_model.to(device)
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()
# all_model_params = list(model.parameters()) + list(encoder.parameters())
# all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
# opt = optim.Adam(lr=0.001, params=all_model_params)

 
def downscale_obs(obs, new_size=(42,42), to_gray=True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else:
        return resize(obs, new_size, anti_aliasing=True)

def prepare_state(state):
    return T.from_numpy(downscale_obs(state, to_gray=True)).float().unsqueeze(dim=0)

def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    tmp = T.from_numpy(downscale_obs(state2, to_gray=True)).float()
    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1

def prepare_initial_state(state,N=3):
    state_ = T.from_numpy(downscale_obs(state, to_gray=True)).float()
    tmp = state_.repeat((N,1,1))
    return tmp.unsqueeze(dim=0)

def policy(qvalues, eps=None):
    if eps is not None:
        if T.rand(1) < eps:
            return T.randint(low=0,high=7,size=(1,))
        else:
            return T.argmax(qvalues, dim=0)
    else:
        return T.multinomial(F.softmax(F.normalize(qvalues), dim=1), num_samples=1)

def loss_fn(q_loss, inverse_loss, forward_loss):
    loss_ = (1 - params['beta']) * inverse_loss
    loss_ += params['beta'] * forward_loss
    loss_ = loss_.sum() / loss_.flatten().shape[0]
    loss = loss_ + params['lambda'] * q_loss
    return loss
 
def reset_env():
    """
    Reset the environment and return a new initial state
    """
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1

def ICM(state1, action, state2, forward_scale=1., inverse_scale=1e4):
    state1_hat = encoder(state1)
    state2_hat = encoder(state2)
    state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
    forward_pred_err = forward_scale * forward_loss(state2_hat_pred,
                       state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat)
    inverse_pred_err = inverse_scale * inverse_loss(pred_action,
                       action.detach().flatten()).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err

def minibatch_train(use_extrinsic=True):
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
    forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch)
    i_reward = (1. / params['eta']) * forward_pred_err
    reward = i_reward.detach()
    if use_extrinsic:
        reward += reward_batch 
    qvals = model(state2_batch)
    print(qvals.is_cuda)
    reward += params['gamma'] * T.max(qvals)
    reward_pred = model(state1_batch)
    reward_target = reward_pred.clone()
    indices = T.stack( (T.arange(action_batch.shape[0]), \
    action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), \
    F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss

epochs = 3500
env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
eps=0.15
losses = []
episode_length = 0
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.
############################################################## find the proper way to change this
# last_x_pos = env.player.prev_x
##############################################################
ep_lengths = []
use_explicit = False
all_model_params = list(model.parameters()) + list(encoder.parameters())
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
opt = optim.Adam(lr=0.001, params=all_model_params)
for i in range(epochs):
    opt.zero_grad()
    episode_length += 1
    q_val_pred = model(state1)
    if i > switch_to_eps_greedy:
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']):
        state2, e_reward_, done, info = env.step(action)
        ##############################################################
        # last_x_pos = info['x_pos']
        ##############################################################
        if done:
            state1 = reset_env()
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = T.stack(list(state_deque),dim=1)
    replay.add_memory(state1, action, e_reward, state2)
    e_reward = 0
    if episode_length > params['max_episode_len']:
        ##############################################################
        # if (info['x_pos'] - last_x_pos) < params['min_progress']:
        ##############################################################
            done = True
        # else:
            ##############################################################
            # last_x_pos = info['x_pos']
            ##############################################################
    if done:
        ep_lengths.append(info['x_pos'])
        state1 = reset_env()
        ##############################################################
        # last_x_pos = env.env.env._x_position
        ##############################################################
        episode_length = 0
    else:
        state1 = state2
    if len(replay.memory) < params['batch_size']:
        continue
    forward_pred_err, inverse_pred_err, q_loss = minibatch_train(use_extrinsic=False)
    loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err)
    loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
                      inverse_pred_err.flatten().mean())
    losses.append(loss_list)
    loss.backward()
    opt.step()


