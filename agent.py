import numpy as np
from tensorflow.keras.models import load_model
from random import randint, sample
from collections import deque
from utils import build_model

class Agent:
    def __init__(self, n_actions,input_dim, batch_size = 8, fname="dqn_model.h5"):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_dec = 0.996
        self.epsilon_min = 0.01
        self.tau = 0.125
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.model_file = fname
        self.memory = deque(maxlen=20000)
        self.start_time = 0.0
        self.stop_time = 0.1
        self.is_collision = False
        self.is_out_of_road = False
        self.has_entred = False
        self.has_arrived = False
        self.is_on_multilane_road = False
        self.nn = build_model(self.learning_rate, n_actions, input_dim, 256, 256)
        self.target_nn = build_model(self.learning_rate, n_actions, input_dim, 256, 256)

    def choose_action(self, state):
        self.epsilon *= self.epsilon_dec
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return randint(0, self.n_actions-1)
        predictions = self.nn.predict(state.flatten())
        return np.argmax(predictions[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = sample(self.memory, self.batch_size)
        for _sample in samples:
            state, action, reward, new_state, done = _sample
            target = self.target_nn.predict(state.flatten())
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_nn.predict(new_state.flatten())[0])
                target[0][action] = reward + Q_future * self.gamma
            self.nn.fit(state.flatten(), target, epochs=1, verbose=0)
    
    def target_train(self):
        weights = self.nn.get_weights()
        target_weights = self.target_nn.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_nn.set_weights(target_weights)

    def save_model(self, fn):
        self.nn.save(fn)

    def load_model(self, fn):
        self.nn = load_model(fn)