# Deep Q network using tensorflow2 and keras
# outputs of DQN correspond directly to the actions you can take (they are the q values)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import os
import cv2



MODEL_NAME = "256x2"
REPLAY_MEMORY_SIZE = 50_000 # can use _ in place of commas, sweet
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.999
UPDATE_TARGET_EVERY = 5


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        gives us 9 total movement options (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3: 
            self.move(x=1, y=-1)
        
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):\
        # if no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # if no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # if we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1



class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3) 
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1 #player key in dict
    FOOD_N = 2 #food key in dict
    ENEMY_N = 3 #enemy key in dict

    #the dict!
    d = {1:(255, 175, 0),
         2:(0, 255, 0),
         3:(0, 0, 255)}


    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.size)


        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player-self.enemy)

        return observation


    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        ### MYABE ####
        #enemy.move()
        #food.move()
        ###############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player-self.enemy)


        # rewards
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == self.ENEMY_PENALTY or self.episode_step >=200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300)) # resizing so we can see the agent in all its glory
        cv2.imshow("image", np.array(img)) #show it
        cv2.waitKey(1)


    # for CNN - needs image input
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8) # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N] #sets the food location tile to the correct color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N] #sets the enemy the correct color from the dict
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N] # sets the player color from the dict
        img = Image.fromarray(env, 'RGB') # reading to rgb. Apparently...
        return img

class DQNAgent:
    def __init__(self):

        #initially the model will fit to random values, which is basically useless. so we have two models, and re-update the weights of target model

        # main model - this is trained with .fit every step
        self.model = self.create_model()

        # Target model - this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # to make the batch size greater than just one step, we will use a random sample from the replay memory array to increase stability
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0 # track when we are ready to update the target model


    def create_model(self):
        #conv net
        model = Sequential()

        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state, step):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory < MIN_REPLAY_MEMORY_SIZE):
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = numpy.array([transition[0] for transition in minibatch])/255

        current_qs_list = self.model.predict(current_states)

        new_current_states = numpy.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] #feature sets
        y = [] #action we decide to take

        #calculate the learned value to keep track of q's
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_future_q
            else:
                new_q = reward

            #after you update the q value, you need to re-fit the neural net
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state) # append the current state
            y.append(current_qs) # the current q values

        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        #updating to determine if we want to update target_model yet
        if terminal_state: #if we are at the end
            self.target_update_counter += 1 # update target counter


        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights()) #set the model weights
            self.target_update_counter = 0 #reset the counter

            





    
# Own Tensorboard class - written by 3rd party - tensorboard normally writes every fit, but we don't want that
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    