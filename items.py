import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.dim)
        self.y = self.clamp(y, self.y_min, self.y_max - self.dim)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.dim)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.dim)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

class SamFisher(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(SamFisher, self).__init__(name, x_max, x_min, y_max, y_min)
        self.dim = 16
        self.lifes = 5
        self.picked_flag = False
        self.flag_dropped = False
        # self.flag_dist = np.inf
        # self.base_dist = np.inf
        self.prev_x = self.x
        self.prev_y = self.y
        self.color = (0, 255, 0)
    
    def color_changer(self):
        if self.picked_flag:
            self.color = (128, 0, 128)

    
class Obstacle(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Obstacle, self).__init__(name, x_max, x_min, y_max, y_min)
        self.dim = 16
        self.color = (0, 0, 0)
    
class Flag(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Flag, self).__init__(name, x_max, x_min, y_max, y_min)
        self.dim = 16
        self.color = (0, 0, 255)

class Base(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Base, self).__init__(name, x_max, x_min, y_max, y_min)
        self.dim = 16
        self.color = (0, 85, 255)
