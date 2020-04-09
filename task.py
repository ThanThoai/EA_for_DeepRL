import numpy as np
import gym 
from env import CartPoleEnv


class CartPole:
    
    def __init__(self):
        self.dim = 5 
        self.env = CartPoleEnv()
        
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp( -x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def action(self, observation, gen):
        gen = gen * 10 - 5
        w = gen[: 4]
        b = gen[4] 
        return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)
    
    def fitness(self, gen):
        fitness = 0
        obervation = self.env.reset()
        for t in range(200):
            action = self.action(obervation, gen)
            obervation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break 
        return fitness
    
    def __del__(self):
        self.env.close()