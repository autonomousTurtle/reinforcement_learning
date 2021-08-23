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



MODEL_NAME = "256x2"
REPLAY_MEMORY_SIZE = 50_000 # can use _ in place of commas, sweet
MINIBATCH_SIZE = 64
MIN_REPLAY_MEMORY_SIZE


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
    



