## Capstone Project @Strive
# Reinforcement Learning with Curiosity

I worked on this project to learn Reinforcement Learning. I created a 'Catch the Flag' game with random generated obstacles, and trained an A2C algorithm with an ICM (Intrinsic Curiosity Module).

The AI agent is able to exlpore the gamefield, to find and catch the flag, and get it back to the base.

<img src='https://github.com/alessiorecchia/curiosity_ai/blob/main/gifs/01.gif' width="245" height="245"> <img src='https://github.com/alessiorecchia/curiosity_ai/blob/main/gifs/02.gif' width="245" height="245"> <img src='https://github.com/alessiorecchia/curiosity_ai/blob/main/gifs/03.gif' width="245" height="245"> <img src='https://github.com/alessiorecchia/curiosity_ai/blob/main/gifs/04.gif' width="245" height="245">

### ICM
I was inspired by <a href='https://worldmodels.github.io/' target="_blank" rel="noopener noreferrer">this paper</a> and by <a href='https://pathak22.github.io/noreward-rl/'>this one</a> for the ICM. With the Intrinsic Curiosity Module the agent is able to explore the environment and lern how to solve the task, self-supervised, even with sparse rewards.
