
import numpy as np
import random
#import time
import cv2
from gym import spaces
#import win32api as wapi
import time
import math

'''
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def get_direction():
    keys = key_check()
    direction = None
    if 'W' in keys:
        direction = 1
    elif 'A' in keys:
        direction = 2
    elif 'S' in keys:
        direction = 3
    elif 'D' in keys:
        direction = 4
    else:
        direction = 0

    return direction
'''

class SnakeGame():

    def __init__(self):
        print('Sucessfully imported SnakeGame')
        self.nb_actions = 4
        self.map_size = (15,15)
        pass


    def reset(self):
        
        
        self.map = np.zeros((self.map_size[0],self.map_size[1]),dtype=np.int8)
        self.snake_pos_head = (self.map_size[0]//2,self.map_size[1]//2)
        self.snake_pos_tail = (self.map_size[0]//2,self.map_size[1]//2)
        self.map[self.map_size[0]//2,self.map_size[1]//2] = 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=np.array(self.map_size), dtype=np.uint8)

        self.add_food()
        return self.map

    def add_food(self):
        while 1:
            x=random.randint(0,14)
            y=random.randint(0,14)
            if self.map[x,y] == 0:
                self.map[x][y] = 5
                self.food_pos=[x,y]
                return

    def render(self, mode="human"):
        color_map = [
            [0, 0, 0],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
            [0, 0, 255],
        ]
        cv2.imshow('Snake', cv2.resize(np.array(
            color_map, dtype=np.uint8)[self.map],(500,500),
                                       interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return self.map



    def get_position_from_direction(self, position, direction):
        position = list(position)

        if direction == 1:
            position[0] -= 1
        elif direction == 2:
            position[1] -= 1
        elif direction == 3:
            position[0] += 1
        elif direction == 4:
            position[1] += 1

        if position[0] > self.map.shape[0] - 1:
            position[0] -= self.map.shape[0]
        if position[0] < 0:
            position[0] += self.map.shape[0]
        if position[1] > self.map.shape[1] - 1:
            position[1] -= self.map.shape[1]
        if position[1] < 0:
            position[1] += self.map.shape[1]

        return tuple(position)



        #print(self.map)
        #exit()
    def get_words_for_direction(self, direction):
        if direction == 0:return("up")
        elif direction == 1:return("left")
        elif direction == 2:return("down")
        elif direction == 3:return('right')


    def step(self, action):
        #action=random.choice([1,2,3,4])
        action+=1
        
        # Get direction from current cell
        self.current_position = self.map[self.snake_pos_head]
        self.player_direction = action


        # Check direction player choosed and update if necessary
        if self.player_direction and self.player_direction != self.current_position:
            if self.player_direction != self.current_position+2 and self.player_direction != self.current_position-2:

                self.map[self.snake_pos_head] = self.player_direction
                self.current_position = self.player_direction
                #print(self.get_words_for_direction(self.current_position))


        # get new position from current position and direction
        new_cell_head = self.get_position_from_direction(self.snake_pos_head, self.current_position)

        # Cell is empty - move snake
        if self.map[new_cell_head] == 0:
            # Move head
            distOld = math.sqrt((self.snake_pos_head[0] - self.food_pos[0])**2 + (self.snake_pos_head[1] - self.food_pos[1])**2) 
            distNew = math.sqrt((new_cell_head[0] - self.food_pos[0])**2 + (new_cell_head[1] - self.food_pos[1])**2) 
            if distNew < distOld:
                reward = 1
            else:
                reward = -3
            self.map[new_cell_head] = self.current_position
            self.snake_pos_head = new_cell_head
            # Get new tail position and move it
            tail_direction = self.map[self.snake_pos_tail]
            new_cell_tail = self.get_position_from_direction(self.snake_pos_tail, tail_direction)
            self.map[self.snake_pos_tail] = 0
            self.snake_pos_tail = new_cell_tail

            return np.array(self.map), reward, False, {}

        # Food
        elif self.map[new_cell_head] == 5:
            # Move head
            self.map[new_cell_head] = self.current_position
            self.snake_pos_head = new_cell_head
            # Do not move tail

            # Add new food
            self.add_food()
            #print(self.map)
            return np.array(self.map), 100, False, {}


        else:
            #print("Game Over")
            #print(self.map)
            return np.array(self.map), -1000, True, {}



    def start(self):

        self.reset()

        while True:

            #action = get_direction()
            _ = self.step(action)
            # print(rew,done, np.sum(obs))
            self.render()
            #time.sleep(0.05)
            if(done):
                self.reset()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

