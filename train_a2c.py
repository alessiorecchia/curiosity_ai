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

try:
    mp.set_start_method('spawn')
except:
    pass

env = GameField()
env.reset()
obs = env.observation()
input_size = np.prod(obs.shape)
hidden_size = 256
n_actions = env.action_space.n


MAX_WORKERS = mp.cpu_count()
n_workers = MAX_WORKERS - 11




model = ActorCritic(input_size=input_size, hidden_size=hidden_size, num_layers=n_actions, n_actions=n_actions)
model.to(device)
model.share_memory()


processes = []
params = {
    'episodes': 5,
    'gamma': 0.95,
    'n_steps': 5,
    'clc': 0.1,
    'max_steps': 100
}
worker = Worker(ac_model=model, env=env, params=params)
# worker.run_worker()


if __name__ == '__main__':

    counter = mp.Value('i',0)
    for i in range(n_workers):
        p = mp.Process(target=worker.run_worker, args=(i,counter))
        p.start() 
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    
    
    print(counter.value,processes[1].exitcode)