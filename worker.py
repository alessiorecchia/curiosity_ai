from os import wait
import torch as T
import torch.multiprocessing as mp
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F


'''
def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env,worker_model)
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards)
        counter.value = counter.value + 1
'''




class Worker():
    def __init__(self, model, env, counter, **kwargs) -> None:
        self.model = model
        self.env = env
        self.counter = counter
        self.episodes = kwargs['episodes']
        self.optimizer = optim.Adam(lr=1e-4, params=self.model.parameters())
        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']
        self.clc = kwargs['clc']
    
    def run_worker(self):
        for i in range(self.episodes):
            self.optimizer.zero_grad()
            self.values, self.log_probs, self.rewards = self.run_episode(self.env, self.model)
            self.actor_loss, self.critic_loss, self.ep_len = self.update_params(self.optimizer, self.values,
                                                                           self.log_probs, self.rewards)
            self.counter.value +=1

    def run_episode(self):
        obs = self.env.observation()
        values, log_probs, rewards = [], [], []
        done = False
        j = 0
        self.R = T.tensor([0])
        while not done and j < self.n_steps:
            j += 1
            action, scores, value = self.model(obs)
            values.append(value)
            policy = F.log_softmax(scores, dim=0)
            log_prob = policy[action]
            log_probs.append[log_prob]
            obs, _, done, info = self.env.step(action.detach().numpy())
            if done:
                reward = -10
                self.env.reset()
            else:
                reward = 1.0
                self.R = value.detach()
            rewards.append(reward)

        return values, log_probs, rewards, self.R

    def update_parameters(self):
        self.rewards = self.rewards.flip(dims=(0,)).view(-1)
        self.log_probs = T.stack(self.log_probs).flip(dims=(0,)).view(-1)
        self.values = T.stack(self.values).flip(dims=(0,)).view-1
        returns = []
        ret_ = self.R
        for i in range(self.rewards.shape[0]):
            ret_ = self.rewards[i] + self.gamma * ret_
            returns.append(ret_)
        returns = T.stack(returns.view(-1))
        returns = F.normalize(returns, dim=0)
        self.actor_loss = -1 * self.log_probs * (returns - self.values.detach())
        self.critic_loss = T.pow(self.values - returns, 2)
        self.loss = self.actor_loss.sum() + self.clc * self.critic_loss.sum()
        self.loss.backward()
        self.optimizer.step()
        return self.actor_loss, self.critic_loss, len(self.rewards)



