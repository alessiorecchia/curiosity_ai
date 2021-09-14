import torch as T
import torch.multiprocessing as mp
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F

from curiosity import Encoder_mod, Forward_mod, Inverse_mod

import logging
from time import time
from socket import gethostname

logger = logging.getLogger('Curiosity AI')
logging.basicConfig(filename='logs/training.log',
                        filemode='a',
                        format='%(asctime)s | %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

HOSTNAME = gethostname()
TARGET_HOST = 'bridgestone'

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Worker():
    def __init__(self, ac_model, icm, env, params) -> None:
        # kwargs = kwargs['kwargs']
        self.ac_model = ac_model
        self.forward_model = icm.forward_model
        self.inverse_model = icm.inverse_model
        self.encoder = icm.encoder
        self.icm = icm
        self.env = env
        self.max_ep_steps = params['max_steps']
        self.episodes = params['episodes']
        self.n_steps = params['n_steps']
        self.gamma = params['gamma']
        self.clc = params['clc']
        self.eta = params['eta']
        self.use_extrinsic = params['use_extrinsic']
        self.beta = params['beta']
        self.lambda_ = params['lambda']
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=1e-4)
    
    def run_worker(self, t, counter):
        # self.counter = counter
        self.t = t
        for i in range(self.episodes):
            steps = 0
            episode_start = time()
            self.done = False
            # self.env.reset()
            self.optimizer.zero_grad()
            while not self.done and steps < self.max_ep_steps:
                self.optimizer.zero_grad()
                self.values, self.log_probs, self.rewards, self.R = self.run_episode()
                self.actor_loss, self.critic_loss, self.ep_len = self.update_parameters()
                steps += 1
                counter.value +=1
                if HOSTNAME != TARGET_HOST:
                    self.env.render()
            if HOSTNAME != TARGET_HOST:
                print(f'Episode: {i} | Episode lenght: {steps} | Time elapsed: {time() - episode_start} | Loss: {self.loss.item()}')
            else:
                logger.info(f'Episode: {i} | Episode lenght: {steps} | Time elapsed: {time() - episode_start} | Loss: {self.loss.item()}')
            if i % 50 == 0:
                T.save(self.ac_model.state_dict(), f'models/ac_model_checkpoint_{i}')
        print('Training done!')

    def run_episode(self):
        obs = self.env.observation()
        values, log_probs, rewards = [], [], []
        j = 0
        self.R = T.tensor([0])
        while not self.done and j < self.n_steps:

            j += 1

            action, scores, value = self.ac_model(obs)
            values.append(value)
            policy = F.log_softmax(scores, dim=0)
            log_prob = policy[action.item()]
            log_probs.append(log_prob)
            obs_next, e_reward, self.done, info = self.env.step(action.detach().item())
            self.forward_pred_err, self.inverse_pred_err = self.icm.predict(action, obs, obs_next)
            reward_ = (1. / self.eta) * self.forward_pred_err
            reward = reward_.detach()
            if self.use_extrinsic:
                # e_reward = e_reward.to(device)
                reward += e_reward
            if self.done:
                reward += 10
                self.env.reset()
            else:
                self.R = value.detach()
            rewards.append(reward)
            obs = obs_next
        # print(f'Done: {self.done} | Early stop: {j < self.n_steps} | j: {j}')
        return values, log_probs, rewards, self.R

    def update_parameters(self):
        self.rewards = T.tensor(self.rewards).flip(dims=(0,)).view(-1)
        self.log_probs = T.stack(self.log_probs).flip(dims=(0,)).view(-1)
        self.values = T.stack(self.values).flip(dims=(0,)).view(-1)
        returns = []
        ret_ = self.R
        for i in range(self.rewards.shape[0]):
            ret_ = self.rewards[i] + self.gamma * ret_
            returns.append(ret_)
        returns = T.stack(returns).view(-1)
        returns = F.normalize(returns, dim=0)
        self.actor_loss = -1 * self.log_probs * (returns - self.values.detach())
        self.critic_loss = T.pow(self.values - returns, 2)
        self.ac_loss = self.actor_loss.sum() + self.clc * self.critic_loss.sum()
        loss_ = (1 - self.beta) * self.icm.inverse_loss + self.beta * self.icm.forward_loss
        self.loss = loss_ + self.lambda_ * self.ac_loss
        self.loss.backward()
        self.optimizer.step()
        # print('Update done')
        return self.actor_loss, self.critic_loss, len(self.rewards)
    
    '''
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
    '''