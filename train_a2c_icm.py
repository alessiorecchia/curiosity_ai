import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from time import time

from gameplay import GameField
from actor_critic import ActorCritic
from curiosity import ICM


import logging
import sys
from socket import gethostname
from glob import glob

device = T.device("cuda" if T.cuda.is_available() else "cpu")

params = {
    'load_last': False,
    'episodes': 500,
    'n_steps': 10,
    'clc': 0.1,
    'max_steps': 3000,
    'use_extrinsic' : True,
    'eta' : 1.5, # scales intrinsic reward
    'gamma': 0.95, # discounts future returns
    'beta' : 0.2, # factor to combine forward and inverse losses in the icm loss
    'lambda_' : 0.2 # scales AC loss in total loss
}

try:
    TARGET_HOST = sys.argv[1]
except:
    TARGET_HOST = 'bridgestone'

HOST = gethostname()
MODE = 'rgb_array' if HOST == TARGET_HOST else 'human'
MODEL_PATH = 'models/'
START_EPISODE = 0
REC_INTERVAL = 10
SAVE_INTERVAL = 10

logger = logging.getLogger('Curiosity AI')
logging.basicConfig(filename='logs/training.log',
                        filemode='a',
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


def get_latest_checkpoint():
    cp_list = glob(MODEL_PATH + '*')
    latest_file = max(cp_list, key=os.path.getctime)
    return latest_file.split('_')[-1]

env = GameField()
env.reset()
action_dict = env.get_action_meanings()
obs = env.observation()

input_size = np.prod(obs.shape)
hidden_size = 256
n_actions = env.action_space.n
icm_in_shape = np.expand_dims(obs, axis=0).shape

ac_model = ActorCritic(input_size=input_size, hidden_size=hidden_size, num_layers=n_actions, n_actions=n_actions)
icm = ICM(icm_in_shape, n_actions, forward_scale=1., inverse_scale=10)

if params['load_last']:
    last = get_latest_checkpoint()
    ac_model.load_state_dict(T.load(MODEL_PATH + f'ac_model_checkpoint_{last}'))
    icm.load_state_dict(T.load(MODEL_PATH + f'icm_checkpoint_{last}'))
    START_EPISODE = int(last)
    params['load_last'] = False

ac_model.to(device)
icm.to(device)

optimizer = optim.Adam(ac_model.parameters(), lr=3e-3)
scheduler = ExponentialLR(optimizer, 0.95)

if __name__ == '__main__':

    for episode in range(START_EPISODE, params['episodes']):

        episode_start = time()
        steps = 0
        done = 0
        record_video = episode % REC_INTERVAL == 0

############################## code for record an episode video ########################################################
        if HOST == TARGET_HOST and record_video:
            size = env.env_shape
            fps = 120
            out = cv2.VideoWriter(f'video/{episode}_output.avi', cv2.VideoWriter_fourcc(*'X264'), fps, (size[0], size[1]))
############################## code for record an episode video ########################################################

        while not done and steps <= params['max_steps']:

            obs = env.observation()
            values, log_probs, rewards, fw_pred_err, inv_pred_err = [], [], [], [], []
            j = 0
            R = T.tensor([0])
            optimizer.zero_grad()
            while not done and j < params['n_steps']:

                j += 1

                action, scores, value = ac_model(obs)
                values.append(value)
                policy = F.log_softmax(scores, dim=0)
                log_prob = policy[action.item()]
                log_probs.append(log_prob)
                obs_next, e_reward, done, info = env.step(action.detach().item())

                forward_pred_err, inverse_pred_err = icm.predict(action, obs, obs_next)
                fw_pred_err.append(forward_pred_err)
                inv_pred_err.append(inverse_pred_err)

                reward_ = params['eta'] * forward_pred_err
                reward = reward_.detach()
                if params['use_extrinsic']:
                    reward += e_reward
                if done:
                    reward += 10
                    env.reset()
                else:
                    R = value.detach()
                rewards.append(reward)
                _ = env.render_obs(mode=MODE)
                obs = obs_next
                env.render_info = (episode, steps, action_dict[action.item()])
                frame = env.render(mode=MODE)

############################## code for record an episode video ########################################################
                # code for record an episode video
                if HOST == TARGET_HOST and record_video:
                    out.write(np.uint8(frame))
############################## code for record an episode video ########################################################
            
            rewards = T.tensor(rewards).flip(dims=(0,)).view(-1)
            log_probs = T.stack(log_probs).flip(dims=(0,)).view(-1)
            values = T.stack(values).flip(dims=(0,)).view(-1)
            fw_pred_errs = T.stack(fw_pred_err).flip(dims=(0,)).view(-1)
            inv_pred_errs = T.stack(inv_pred_err).flip(dims=(0,)).view(-1)


            returns = []
            ret_ = R
            for i in range(rewards.shape[0]):
                ret_ = rewards[i] + params['gamma'] * ret_
                returns.append(ret_)
            returns = T.stack(returns).view(-1)
            returns = F.normalize(returns, dim=0)
            actor_loss = -1 * log_probs * (returns - values.detach())
            critic_loss = T.pow(values - returns, 2)
            ac_loss = actor_loss.sum() + params['clc'] * critic_loss.sum()

            icm_loss = (1 - params['beta']) * fw_pred_errs + params['beta'] * inv_pred_errs
            icm_loss = icm_loss.sum()/icm_loss.shape[0]
            loss = icm_loss + params['lambda_'] * ac_loss
            loss.backward()
            optimizer.step()

            steps += 1
            if fw_pred_err:
                env.footer_info = (forward_pred_err.item(), reward.item(), action_dict[icm.action_hat.item()])
        
        if HOST == TARGET_HOST:

############################## code for record an episode video ########################################################
            if record_video:
                out.release()
############################## code for record an episode video ########################################################

            logger.info(f'Episode {episode} |  Episode lenght: {steps} | Time elapsed: {int(time() - episode_start)} | Loss: {loss.item()} | AC Loss: {ac_loss.item()} | FW Loss: {forward_pred_err.item()} | Inv Loss: {inverse_pred_err.item()}')
        else:
            print(f'Episode {episode} \n \
                    Episode lenght: {steps} \n \
                    Time elapsed: {int(time() - episode_start)} \n \
                    Loss: {loss.item()} \n \
                    AC Loss: {ac_loss.item()} \n \
                    FW Loss: {forward_pred_err.item()} \n \
                    Inv Loss: {inverse_pred_err.item()}')
        
        if episode % SAVE_INTERVAL == 0:
                T.save(ac_model.state_dict(), f'models/ac_model_checkpoint_{episode}')
                T.save(icm.state_dict(), f'models/icm_checkpoint_{episode}')
        
        env.reset()
    
