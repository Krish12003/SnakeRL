#snake_ai_from_scratch.py

#import necessary modules

from collections import deque

import numpy as np
import random
from tqdm import tqdm
import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv2D, Flatten
from keras.optimizers import Adam


from snake_env import SnakeGame

#initalilze game

class SnakeDQNAgent:
    def __init__(self, env):
        self.state_size = 15
        self.action_size = 4
        self.nb_actions = env.nb_actions
        self.observation_size = env.map_size
        self.learning_rate = 0.00075
        self.memory = deque(maxlen=15)
        self.epsilon = 0.66
        self.decay = 1
        self.gamma = 0.95
        self.epsilon_min = 0.01 
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(5, kernel_size=(1,1), activation='relu',input_shape=(15,15,1),data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_dim = (15,25,25) , activation = 'relu'))
        model.add(LSTM(256, activation = 'relu'))
        model.add(Dense(self.nb_actions, activation = 'linear'))
        model.compile(loss='mse', optimizer = Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #print(state.shape)
        act_values = self.model.predict(state.reshape(-1,15,15,1))
        return np.argmax(act_values[0]) 
    def save(self, name):
        self.model.save(f'snake_{name}eps')

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #print(len(minibatch))
        for state in minibatch:
            #print(state[0])
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(-1,15,15,1))[0]))
            target_f = self.model.predict(state[0].reshape(-1,15,15,1))
            target_f[0][action] = target
            self.model.fit(state[0].reshape(-1,15,15,1), target_f, epochs=1, verbose=0)


episodes = 1000

env = SnakeGame()
agent = SnakeDQNAgent(env)
batch_size = 8
for ep in range(episodes):
    print(ep)
    state = env.reset()
    for time in range(1000):
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        #agent.remember(next_state)
        state=next_state
        if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(ep, episodes, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if ep % 25 == 0:
        agent.save(ep)





