import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
from items import Obstacle, Flag, Base, SamFisher

from gym import Env, spaces
import time
import socket


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

N_OBS = 32
# MIN_CONT_OBS = 2
# MAX_CONT_OBS = 5
D_MOVE = 3
HOSTNAME = socket.gethostname()

class GameField(Env):
    def __init__(self):
        super(GameField, self).__init__()

        self.render_info = None
        self.footer_info = None
        
        # Define a 2-D observation space
        self.observation_shape = (500, 500, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
    
        # Define an action space ranging from 0 to 5
        self.action_space = spaces.Discrete(6,)
                        
        # Create a canvas to render the environment images upon 
        self.canvas = np.zeros(self.observation_shape)
        self.pad = 70
        
        
        # Define elements present inside the environment
        self.elements = []

        # Permissible area of sam_fisher to be 
        # self.y_min = int (self.observation_shape[0] * 0.05)
        self.y_min = int ((self.observation_shape[0] - 2 * self.pad) * 0.05 + self.pad)
        self.x_min = int ((self.observation_shape[1] - 2 * self.pad) * 0.05 + self.pad)
        self.y_max = int ((self.observation_shape[0] - 2 * self.pad) * 0.95 + self.pad)
        self.x_max = int ((self.observation_shape[1] - 2 * self.pad) * 0.95 + self.pad)

    def draw_elements_on_canvas(self, render_info=None, footer_info=None):
        # Init the canvas 
        self.canvas = np.zeros(self.observation_shape)
        cv2.rectangle(self.canvas, (self.pad, self.pad),
                     (self.observation_shape[1]-self.pad, self.observation_shape[0]-self.pad),
                     (255, 255, 255), -1)

        lifes = 0

        # Draw the sam_fisher, obstacles, Base and flag on canvas

        # test code 
        ##############################################################################################################
        for elem in self.elements:
            x,y = elem.x, elem.y
            # elem.icon = cv2.addWeighted(self.canvas[y: y + elem_shape[1], x:x + elem_shape[0]],0.4,elem.icon,0.1,0)
            # self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon
            if type(elem).__name__ == 'Obstacle':
                cv2.rectangle(self.canvas, (x-elem.dim//2, y-elem.dim//2), (x+elem.dim//2, y+elem.dim//2), elem.color, -1)
            
            if type(elem).__name__ == 'Flag':
                cv2.circle(self.canvas,(x, y), elem.dim//2 , elem.color, -1) 


            if type(elem).__name__ == 'SamFisher':
                elem.color_changer()
                cv2.circle(self.canvas,(x, y), elem.dim//2, elem.color, -1)
                lifes = elem.lifes
            
            if type(elem).__name__ == 'Base':
                cv2.circle(self.canvas,(x, y), elem.dim , elem.color, 4)

            
        ##############################################################################################################
        if render_info:
            text = 'Episode: {} | steps: {} | Action: {}'.format(render_info[0], render_info[1], render_info[2])
            self.canvas = cv2.putText(self.canvas, text, (10,20), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        if footer_info:
            footer = f'FW Loss: {footer_info[0]:.2f} | Reward: {footer_info[1]:.2f} | Advantage'
            self.canvas = cv2.putText(self.canvas, footer, (10,490), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    
    def reset(self):

        # reset obstacles
        self.obs_count = 0

        # Reset the reward
        self.ep_return  = 0

        # Reset elemnts
        self.elements = []

        def assing_y(x, limit):
            if self.x_min + limit < x < self.x_max - limit:
                return random.choice([random.randrange(self.y_min, self.y_min + limit),
                                   random.randrange(self.y_max - limit, self.y_max)])
            else:
                return random.randrange(self.y_min, self.y_max)

        # Creating a random map
        for i in range(N_OBS):
            obstacle = Obstacle(f'obstacle_{self.obs_count}', self.x_max, self.x_min, self.y_max, self.y_min)
            self.obs_count += 1
            boundary = 20

            obs_x = random.randrange(self.x_min + boundary, self.x_max - boundary)
            obs_y = random.randrange(self.y_min + boundary, self.y_max - boundary)

            obstacle.set_position(obs_x, obs_y)
            self.elements.append(obstacle)
            


        # Determine a place to intialize the Base and SamFisher in
        base_not_initialized = True

        # x = random.randrange(self.x_min, self.x_max)
        # y = assing_y(x, 20)
        x = self.x_min
        y = (self.y_min + self.y_max) // 2

        # Determine a place to intialize the Flag in
        # x_flag = random.randrange(self.x_min, self.x_max)
        # y_flag = assing_y(x_flag, 20)
        x_flag = self.x_max
        y_flag = (self.y_min + self.y_max) // 2
        
        # Intialize the SamFisher, Base and Flag
        self.base = Base("base", self.x_max, self.x_min, self.y_max, self.y_min)
        self.base.set_position(x,y)
        self.elements.append(self.base)

        self.player = SamFisher("sam_fisher", self.x_max, self.x_min, self.y_max, self.y_min)
        self.player.set_position(x, y)
        self.elements.append(self.player)

        self.flag = Flag("flag", self.x_max, self.x_min, self.y_max, self.y_min)
        self.flag.set_position(x_flag, y_flag)
        self.elements.append(self.flag)

        # Reset the Canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas(self.render_info, self.footer_info)

        # return the observation
        return self.canvas
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(1)
        
        elif mode == "rgb_array":
            return self.canvas

    def render_obs(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            frame = self.observation()
            cv2.imshow("obs", frame)
            cv2.waitKey(1)
        
        elif mode == "rgb_array":
            return self.canvas
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Pickup flag", 5: "Drop flag"}
    
    def get_dist(self, elem1, elem2):
        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()
        return np.sqrt((elem1_x-elem2_x)**2 + (elem1_y - elem2_y)**2)

    
    def has_collided(self, elem1, elem2):
        

        dist = self.get_dist(elem1, elem2)
        
        if type(elem2).__name__ == 'Obstacle':
            if dist <= elem1.dim/2 + elem2.dim/2:
                return True
        else:
            if dist <= max(elem1.dim, elem2.dim) - 2:
                return True

        return False


    # def good_direction(self, p):
    #     gd = False
    #     # flag_dist = np.sqrt((p.x - p.x)**2 + (p.y - f.y)**2)
    #     if p.picked_flag == False:
    #         gd = True
    #     return gd
    
    def wall_collision(self, elements):

        collision = False

        # For elements in the Ev
        for elem in elements:
            if isinstance(elem, Obstacle):
                
                # Sam tried to pass throgh a wall.
                if self.has_collided(self.player, elem):
                    collision = True
                    self.player.x = self.player.prev_x
                    self.player.y = self.player.prev_y
        return collision
    
    def observation(self):
        x, y = self.player.get_position()
        pad = self.pad - 1
        screen = self.canvas
        screen = np.float32(screen)
        # print(screen.shape)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # print(screen.shape)
        screen = screen[y-pad:y+pad, x-pad:x+pad]
        screen = cv2.resize(screen, (21, 21), interpolation=cv2.INTER_CUBIC)
        # screen = screen.transpose((2, 0, 1))
        return screen / 255

    
    
    def step(self, action):

        # self.player.flag_dist = self.get_dist(self.player, self.flag)
        # self.player.base_dist = self.get_dist(self.player, self.base)

        self.player.prev_x = self.player.x
        self.player.prev_y = self.player.y

        info = {}


        # Flag that marks the termination of an episode
        done = False
        reward = 0
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"
        
        # Reward for executing a step.

        # changed to 0
        # reward =  -1 # default reward if there is no flag pickup or drop


        # apply the action to Sam Fisher {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Pickup flag", 5: "Drop flag"}
        

        if action == 0:             # move right
            self.player.move(D_MOVE, 0)
            self.wall_collision(self.elements)
            # reward = -2
            # if self.player.picked_flag:
            #     if (self.player.x - self.base.x)**2 < (self.player.prev_x - self.base.x)**2:
            #         reward = -1
            # else:
            #     if (self.player.x - self.flag.x)**2 < (self.player.prev_x - self.flag.x)**2:
            #         reward = -1
            # if self.wall_collision(self.elements):
            #     reward = -10
                            
             
        elif action == 1:           # move left
            self.player.move(-1 * D_MOVE, 0)
            self.wall_collision(self.elements)
            # reward = -2
            # if self.player.picked_flag:
            #     if (self.player.x - self.base.x)**2 < (self.player.prev_x - self.base.x)**2:
            #         reward = -1
            # else:
            #     if (self.player.x - self.flag.x)**2 < (self.player.prev_x - self.flag.x)**2:
            #         reward = -1
            # if self.wall_collision(self.elements):
            #     reward = -10
            
        elif action == 2:           # move down
            self.player.move(0, D_MOVE)
            self.wall_collision(self.elements)
            # reward = -2
            # if self.player.picked_flag:
            #     if (self.player.y - self.base.y)**2 < (self.player.prev_y - self.base.y)**2:
            #         reward = -1
            # else:
            #     if (self.player.y - self.flag.y)**2 < (self.player.prev_y - self.flag.y)**2:
            #         reward = -1
            # if self.wall_collision(self.elements):
            #     reward = -10
             
        elif action == 3:           # move up
            self.player.move(0, -1 * D_MOVE)
            self.wall_collision(self.elements)
            # reward = -2
            # if self.player.picked_flag:
            #     if (self.player.y - self.base.y)**2 < (self.player.prev_y - self.base.y)**2:
            #         reward = -1
            # else:
            #     if (self.player.y - self.flag.y)**2 < (self.player.prev_y - self.flag.y)**2:
            #         reward = -1
            # if self.wall_collision(self.elements):
            #     reward = -10
            
        elif action == 4:           # pickup flag
            if self.has_collided(self.player, self.flag):
                if self.flag in self.elements:
                    self.elements.remove(self.flag)
                    self.player.picked_flag = True
                    reward = 20
            # else:
            #     reward = -15
                
            # elif self.wall_collision(self.elements):
            #     reward = -10

        elif action == 5:           # drop flag
            if self.has_collided(self.player, self.base):
                if self.player.picked_flag == True:
                    self.player.flag_dropped = True
                    done = True
                    reward = 30
            # else:
                # reward = -15
            # elif self.wall_collision(self.elements):
            #     reward = -10




        # Draw elements on the canvas
        self.draw_elements_on_canvas(self.render_info, self.footer_info)

        # return self.canvas, reward, done, []
        return self.observation(), reward, done, info


# env = GameField()
# print(env.x_min, env.x_max, env.y_min, env.y_max)
# env.reset()

# obs = env.observation()
# # obs = obs.transpose(1, 2, 0)


# plt.imshow(obs)
# plt.show()

# # plt.imshow(env.elements[0].icon)
# plt.show()
# print(HOSTNAME)