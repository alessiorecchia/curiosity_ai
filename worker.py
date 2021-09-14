import torch as T
import torch.multiprocessing as mp
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F

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

class Worker():
    def __init__(self, ac_model, env, params) -> None:
        # kwargs = kwargs['kwargs']
        self.ac_model = ac_model
        self.env = env
        self.max_ep_steps = params['max_steps']
        self.episodes = params['episodes']
        self.n_steps = params['n_steps']
        self.gamma = params['gamma']
        self.clc = params['clc']
        self.ac_optimizer = optim.Adam(self.ac_model.parameters(), lr=1e-4)
    
    def run_worker(self, t, counter):
        # self.counter = counter
        self.t = t
        for i in range(self.episodes):
            steps = 0
            episode_start = time()
            self.done = False
            self.env.reset()
            self.ac_optimizer.zero_grad()
            while not self.done and steps < self.max_ep_steps:
                self.ac_optimizer.zero_grad()
                self.values, self.log_probs, self.rewards, self.R = self.run_episode()
                self.actor_loss, self.critic_loss, self.ep_len = self.update_parameters()
                steps += 1
                counter.value +=1
                if HOSTNAME != TARGET_HOST:
                    self.env.render()
            if HOSTNAME != TARGET_HOST:
                print(f'Episode: {i} | Episode lenght: {steps} | Time elapsed: {time() - episode_start} | Loss: {self.loss}')
            else:
                logger.info(f'Episode: {i} | Episode lenght: {steps} | Time elapsed: {time() - episode_start} | Loss: {self.loss}')
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
            obs, reward, self.done, info = self.env.step(action.detach().item())
            if self.done:
                reward += 10
                self.env.reset()
            else:
                self.R = value.detach()
            rewards.append(reward)
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
        self.loss = self.actor_loss.sum() + self.clc * self.critic_loss.sum()
        self.loss.backward()
        self.ac_optimizer.step()
        # print('Update done')
        return self.actor_loss, self.critic_loss, len(self.rewards)