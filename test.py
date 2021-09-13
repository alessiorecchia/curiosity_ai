import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2

from worker import Worker
from gameplay import GameField
from actor_critic import ActorCritic

# print(mp.cpu_count())

device = T.device("cuda" if T.cuda.is_available() else "cpu")

env = GameField()
env.reset()
obs = env.observation()
input_size = np.prod(obs.shape)
hidden_size = 256
n_actions = env.action_space.n

model = ActorCritic(input_size=input_size, hidden_size=hidden_size, num_layers=n_actions, n_actions=n_actions)
model.to(device)
params = {
    'episodes': 5,
    'gamma': 0.95,
    'n_steps': 10,
    'clc': 0.1,
    'max_steps': 100
}
worker = Worker(t=1, model=model, env=env, counter=1, params=params)
worker.run_worker()