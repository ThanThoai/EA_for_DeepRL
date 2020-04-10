import gym
import numpy as np
import random
env = gym.make("CartPole-v1")
shape_of_net = (4,
                32,
                16,
                2)
num_layers = len(shape_of_net) - 1

def softmax(x):
    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Agent:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.cum_reward = 0
        for i in range(num_layers):
            self.weights.append(np.random.uniform(-3,3,size=(shape_of_net[i],shape_of_net[i+1])))
            self.biases.append(np.random.uniform(-3,3,size=(shape_of_net[i+1])))
    
    def act(self,state):
        l = state
        for i in range(num_layers):
            l = np.matmul(l,self.weights[i]) + self.biases[i]
            if i != num_layers-1:
                #ReLU
                np.maximum(l,0,l)
        
        l = softmax(l)
        
        return np.argmax(l[0])

class Population:
    
    def __init__(self, num_agent, num_gen, top_perc):
        self.num_agent = num_agent
        self.num_gen = num_gen
        self.cutoff = int(top_perc * num_agent)
        
    def init_pop(self):
        pop = []
        for _ in range(self.num_agent):
            pop.append(Agent())
        return pop
    
    def crossover(self, parents, offsprings):
        for offspring in offsprings:
            parent1,parent2 = random.sample(parents,2)
            for i in range(num_layers):
                offspring.weights[i] = np.concatenate([parent1.weights[i][:,:int(shape_of_net[i+1]/2)],
                                                           parent2.weights[i][:,int(shape_of_net[i+1]/2):]],axis=1)
                offspring.biases[i] = np.concatenate([parent1.biases[i][:int(shape_of_net[i+1]/2)],parent2.biases[i][int(shape_of_net[i+1]/2):]])
    
    def mutation(self, pop):
        for agent in pop:
            for i in range(num_layers):
                #Mutate 
                mutation = np.random.normal(scale=3,size=2)
                agent.weights[i][np.random.randint(0,shape_of_net[i]),np.random.randint(0,shape_of_net[i+1])] += mutation[0]
                agent.biases[i][np.random.randint(0,shape_of_net[i+1])] += mutation[1]
                
    def run(self):
        pop = self.init_pop()
        for i in range(self.num_gen):
            for agent in pop:
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state.reshape((1,-1)))
                    state,r,done,_ = env.step(action)
                    agent.cum_reward += r
            agent_reward_list = sorted(pop,key= lambda a:a.cum_reward,reverse=True)
            self.crossover(agent_reward_list[:self.cutoff],agent_reward_list[self.cutoff:])
            self.mutation(pop)
            print(f"generation {i} avg_score:{sum([a.cum_reward for a in pop])/self.num_agent}")
            for agent in pop:
                agent.cum_reward = 0
        self.res = agent_reward_list[0]
    
    def play(self):
        for _ in range(10):
            done = False
            state = env.reset()
            score = 0
            while not done:
                env.render()
                action = self.res.act(state.reshape((1,-1)))
                state,r,done,_ = env.step(action)
                score += r
            print(score)
        
if __name__ == "__main__":
    app = Population(100, 30, 0.1)
    app.run()
    app.play()