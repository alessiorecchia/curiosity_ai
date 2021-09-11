import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging
from time import time
from socket import gethostname

from skimage.transform import resize
from collections import deque


from curiosity import Encoder_mod, Forward_mod, Inverse_mod
from memory import MemoryReplay
from dqn import DQN
from gameplay import HOSTNAME, GameField

device = T.device("cuda" if T.cuda.is_available() else "cpu")

logger = logging.getLogger('Curiosity AI')
logging.basicConfig(filename='logs/training.log',
                        filemode='a',
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


params = {
    'batch_size':128,
    'beta':0.2,
    'lambda':0.1,
    'eta': 1.0,
    'gamma':0.2,
    'max_episode_len':200,
    # 'min_progress':15,
    'action_repeats':6,
    'frames_per_state':3
}

env = GameField()
n_actions = env.action_space.n
replay = MemoryReplay(N=1000, batch_size=params['batch_size'])
model = DQN(42, 42, n_actions)
model.to(device)
encoder = Encoder_mod()
encoder.to(device)
forward_model = Forward_mod()
forward_model.to(device)
inverse_model = Inverse_mod()
inverse_model.to(device)
forward_loss = nn.MSELoss(reduction='none')
inverse_loss = nn.CrossEntropyLoss(reduction='none')
qloss = nn.MSELoss()

HOSTNAME = gethostname()
TARGET_HOST = 'bridgestone'

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
    # if eps is not None:
    #     if T.rand(1) < eps:
    #         return T.randint(low=0,high=n_actions,size=(1,))
    #     else:
    #         return T.argmax(qvalues, dim=1)
    # else:
    #     return T.multinomial(F.softmax(F.normalize(qvalues), dim=1), num_samples=1)
    if T.rand(1) < eps:
        return T.randint(low=0,high=n_actions,size=(1,))
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
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = reward_batch.view(reward_batch.shape[0],1)
    
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
    indices = T.stack( (T.arange(action_batch.shape[0]), \
    action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), \
    F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss

episodes = 1
env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
eps=1.
eps_decay = 0.995
eps_min = 0.15
losses = []
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.
############################################################## find the proper way to change this
# last_x_pos = env.player.prev_x
##############################################################
ep_lengths = []
use_explicit = True


all_model_params = list(model.parameters()) + list(encoder.parameters())
all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
optimizer = optim.Adam(lr=0.001, params=all_model_params)

start = time()

for episode in range(episodes):
    done = False
    episode_length = 0
    episode_start = time()

    while not done:
        optimizer.zero_grad()
        episode_length += 1
        q_val_pred = model(state1)
        # if episode_length > switch_to_eps_greedy:
        #     # item = policy(q_val_pred,eps=eps)
        #     # print(item)
        #     # print(type(item.item()))
        #     action = int(policy(q_val_pred,eps))
        # else:
        #     action = int(policy(q_val_pred))
        action = int(policy(q_val_pred,eps))
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
            # ep_lengths.append(info['x_pos'])
            ep_lengths.append(episode_length)
            state1 = reset_env()
            ##############################################################
            # last_x_pos = env.env.env._x_position
            ##############################################################
        else:
            state1 = state2
        if len(replay.memory) < params['batch_size']:
            continue
        env.render()
        forward_pred_err, inverse_pred_err, q_loss = minibatch_train(use_extrinsic=use_explicit)
        loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err)
        loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
                        inverse_pred_err.flatten().mean())
        losses.append(loss_list)
        loss.backward()
        optimizer.step()
        if eps > eps_min:
            eps *= eps_decay
        
    if HOSTNAME != TARGET_HOST:
        print(f'Episode: {episode} | Episode lenght: {episode_length} | Time elapsed: {time() - episode_start}')
    else:
        logger.info(f'Episode: {episode} | Episode lenght: {episode_length} | Time elapsed: {time() - episode_start}')
env.close()
losses_cpu = []
for element in losses:
    x, y, z = element
    losses_cpu.append((x.cpu().detach(), y.cpu().detach(), z.cpu().detach()))

if HOSTNAME != TARGET_HOST:
    print(f'Overall time to complete {episodes} episodes: {time() - start}')
else:
    logger.info(f'Overall time to complete {episodes} episodes: {time() - start}')

T.save(model.state_dict(), 'models/dqn')
T.save(forward_model.state_dict(), 'models/fw_icm')
T.save(inverse_model.state_dict(), 'models/inv_icm')
T.save(encoder.state_dict(), 'models/enc_icm')

losses_ = np.array(losses_cpu)
plt.figure(figsize=(8,6))
plt.plot(np.log(losses_[:,0]),label='Q loss')
plt.plot(np.log(losses_[:,1]),label='Forward loss')
plt.plot(np.log(losses_[:,2]),label='Inverse loss')
plt.legend()
# plt.show()
plt.savefig('plots/losses.png')


