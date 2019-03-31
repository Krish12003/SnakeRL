#snake_ai.py

from keras.models import Sequential
from keras.layers import Flatten, Convolution2D, Input, Dense, LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from snake_env import SnakeGame

env = SnakeGame()

nb_actions = 4

INPUT_SHAPE = (15, 15)
WINDOW_LENGTH = 3

input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])

model = Sequential()
model.add(Convolution2D(15, kernel_size=(1,1), activation='relu',input_shape=input_shape))
model.add(Convolution2D(50, kernel_size=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(512,activation='relu', input_shape=input_shape))
model.add(Dense(256,activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=25000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00035), metrics=['mae'])
print('Model Compiled!')


weights_filename = 'dqn_{}_weights.h5f'.format('Snake')
checkpoint_weights_filename = 'dqn_' + 'Snake' + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format('Snake')
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]
print("Starting Training!")
dqn.fit(env, callbacks=callbacks, nb_steps=17500000, log_interval=10000,visualize=True)
dqn.save_weights(weights_filename, overwrite=True)

