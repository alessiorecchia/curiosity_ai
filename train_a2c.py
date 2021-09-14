import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2

from worker import Worker, logger
from gameplay import GameField
from actor_critic import ActorCritic
from curiosity import Forward_mod, Inverse_mod, Encoder_mod, ICM

from socket import gethostname

# print(mp.cpu_count())

device = T.device("cuda" if T.cuda.is_available() else "cpu")
HOST = gethostname()
TARGET_HOST = 'bridgestone'

try:
    mp.set_start_method('spawn')
except:
    pass

env = GameField()
env.reset()
obs = env.observation()
input_size = np.prod(obs.shape)
hidden_size = 128
n_actions = env.action_space.n


MAX_WORKERS = mp.cpu_count()
if HOST == TARGET_HOST:
    n_workers = 8
else:
    n_workers = 2



ac_model = ActorCritic(input_size=input_size, hidden_size=hidden_size, num_layers=n_actions, n_actions=n_actions)
ac_model.to(device)
ac_model.share_memory()
fw_model = Forward_mod(n_actions)
fw_criterion = nn.MSELoss(reduction='none')
fw_model.to(device)
fw_model.share_memory()
inv_model = Inverse_mod(n_actions)
inv_criterion = nn.CrossEntropyLoss(reduction='none')
inv_model.to(device)
inv_model.share_memory()
encoder = Encoder_mod(1)
encoder.to(device)
encoder.share_memory()
icm = ICM(encoder, fw_model, fw_criterion, inv_model, inv_criterion)

processes = []
params = {
    'episodes': 501,
    'gamma': 0.95,
    'n_steps': 5,
    'clc': 0.1,
    'max_steps': 3000,
    'eta' : 1.,
    'use_extrinsic' : True,
    'beta' : 0.2,
    'lambda' : 0.2
}
worker = Worker(ac_model=ac_model, icm=icm, env=env, params=params)
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
    
    if HOST != TARGET_HOST:
        print(counter.value,processes[0].exitcode)
    else:
        logger.info(counter.value,processes[0].exitcode)