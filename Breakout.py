import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import numpy as np
import random
import gym

class StateProcessor:
    def __init__(self):
        -1

    def gray(self, state):
        return np.mean(state, axis=2).astype(np.uint8)


    def crop(self, state):
        return state[::2, ::2]

    def process(self, state):
        return self.gray(self.crop(state))


class ReplayBuffer:
    def __init__(self, max_size):
        self.M = []
        self.max_size = max_size

    def store(self, state, action, reward, next_state, done):
        if self.M == self.max_size:
            self.M.pop(0)
        self.M.append((state, action, reward, next_state, done))

    def batch_sample(self, batch_size):
        batch = random.sample(self.M, batch_size)
        return batch

class Model:
    def __init__(self, n_actions, lr):
        self.NN = self.build_model(n_actions, lr)

    def build_model(self, fc1_dims, n_actions, lr):
        model = Sequential()
        model.add(Conv2D(32, 8, 4, input_shape = (105, 80, 4,), activation = 'relu'))
        model.add(Conv2D(64, 4, 2, activation = 'relu'))
        model.add(Conv2D(64, 3, 1, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dense(n_actions, activation = 'linear'))
        model.compile(optimizer = Adam(lr = lr), loss = 'mse')
        return model

class Agent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.n_actions = 4
        self.lr = 0.0001
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 1 - 1e-5
        self.step = 0
        self.memory_size = 250000
        self.model = Model(self.n_actions, self.lr)
        self.target_model = Model(self.n_actions, self.lr)
        self.memory = ReplayBuffer(self.memory_size)

    def choose_action(self, state):
        state = np.array(state)[np.newaxis, :]
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            actions = self.model.NN.predict(state/255.0)
            return np.argmax(actions[0])

    def update_model(self):
        if self.step % 1000 == 0:
            self.target_model.NN.set_weights(self.model.NN.get_weights())

    def learn(self):
        if self.epsilon > self.epsilon_end:
           self.epsilon *= self.epsilon_decay
        else:
           self.epsilon = self.epsilon_end

        if len(self.memory.M) > self.batch_size:
            batch = self.memory.batch_sample(self.batch_size)
            states = []
            qs = []
            for state, action, reward, next_state, done in batch:
                states.append(state)

                state = np.array(state)[np.newaxis, :]
                next_state = np.array(next_state)[np.newaxis, :]

                if done:
                    target = reward
                if not done:
                    target = reward + self.gamma * np.max(self.target_model.NN.predict(next_state/255.0)[0])

                q = self.model.NN.predict(state/255.0)
                q[0][action] = target

                q = np.squeeze(q)
                qs.append(q)


            self.model.NN.fit(np.array(states)/255.0, np.array(qs), epochs = 1, verbose = 0)

            self.update_model()

            self.step += 1



agent = Agent()
state_processor = StateProcessor()
env = gym.make('BreakoutDeterministic-v4')
env = gym.wrappers.Monitor(env, "./video", video_callable=lambda episode_id: episode_id%1==0,force=True)
env.reset()

n_games = 2000
start = 1

scores = []
for n in range(start, n_games):
    done = False
    score = 0
    state = env.reset()
    state = state_processor.process(state)
    state = np.stack([state] * 4, axis = 2)
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = state_processor.process(next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis = 2)
        score += reward
        agent.memory.store(state, action, reward, next_state, done)
        state = next_state
        agent.learn()

    scores.append(score)

    avg_score = np.mean(scores)
    ten_avg_score = np.mean(scores[-10:])

    print('episode {:4d}   score {:4d}   average score {:4d}     last ten average score {:4d}    epsilon {:4f}    memory {:4d}'.format(n, int(score), int(avg_score), int(ten_avg_score), agent.epsilon, len(agent.memory.M)))
