import numpy as np
import gym 
from env import CartPoleEnv
from typing import List

class CartPole:
    
    def __init__(self, shape_gen : List[int]):
        assert len(shape_gen) >= 2, "Hidden size must >= 1!!!!!!!!!"
        self.shape_gen = shape_gen
        self.param_lst = self.get_param_lst()
        self.dim = len(self.param_lst)
        self.env = CartPoleEnv()
        
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp( -x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def get_param_lst(self):
        param_lst = []
        for i in range(len(self.shape_gen) - 1):
            param_lst.append(self.shape_gen[i] * self.shape_gen[i + 1] + 1)
        return param_lst
            
    def decode(self, gen, G_unified):
        # print(G_unified)
        param = []
        st = 0
        i = 0
        while i < (self.dim - 1):
            w = np.array(gen[st : st + self.param_lst[i] - 1]).reshape((self.shape_gen[i], self.shape_gen[i + 1]))
            b = np.array(gen[st + self.param_lst[i]]).reshape((1, 1))
            st += self.param_lst[i] + 1
            param.append((w, b))
            i += 1

        if self.dim == len(G_unified.keys()):
            w = np.array(gen[st : st + self.param_lst[-1] - 1]).reshape((self.shape_gen[self.dim - 1], self.shape_gen[self.dim]))
            b = np.array(gen[st + self.param_lst[self.dim - 1]]).reshape((1, 1))
            param.append((w, b))
            return param
        else:
            for j in range(i, len(G_unified.keys())):
                if j == len(G_unified) - 1:
                    w = np.array(gen[st : st + self.param_lst[-1] - 1]).reshape((self.shape_gen[self.dim - 1], self.shape_gen[self.dim]))
                    b = np.array(gen[-1]).reshape((1, 1))
                    param.append((w, b))
                    return param
                else:
                    st += G_unified["l%d" %j]
        
    def action(self, observation, gen, G_unified):
        param = self.decode(gen, G_unified)
        (w, b) = param[0]
        Z = self.relu(observation.dot(w) + b)
        for p in param[1: len(param) - 1]:
            (w, b) = p
            Z = self.sigmoid(Z.dot(w) + b)
        (w, p) = param[-1]
        Z = self.sigmoid(Z.dot(w) + b) 
        return int(Z > 0.5)
    
    def fitness(self, gen, G_unified):
        fitness = 0
        obervation = self.env.reset()
        done = False
        while not done:
            action = self.action(obervation, gen, G_unified)
            obervation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break 
        return fitness
    
    def __del__(self):
        self.env.close()