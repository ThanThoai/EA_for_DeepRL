

import numpy as np
import gym 
from task import CartPole

class MFEA:
    def __init__(self, tasks : CartPole, num_pop : int, num_gen : int, sbxdi : float, pmdi : float, rmp : float):
        self.tasks = tasks 
        self.num_task = len(tasks)
        self.num_pop = self.num_task * num_pop
        self.num_gen = num_gen
        self.sbxdi = sbxdi
        self.pmdi = pmdi
        self.rmp = rmp
        self.G_unified, self.num_dim, self.L_unified = self.get_G_unified()
        

    def get_G_unified(self):
        G_unified = {}
        dim = 0
        L_unified = max([task.dim for task in self.tasks])
        for i in range(L_unified):
            G_unified["l%d" %i] = 0
        for l in range(L_unified):
            for k in range(self.num_task):
                if l == self.tasks[k].dim - 1:
                    if self.tasks[k].param_lst[l] > G_unified["l%d" %(L_unified - 1)]:
                        G_unified["l%d" %(L_unified - 1)] = self.tasks[k].param_lst[l]
                elif l < self.tasks[k].dim:
                    if self.tasks[k].param_lst[l] > G_unified["l%d" %l]:
                        G_unified["l%d" %l] = self.tasks[k].param_lst[l]
        for key in G_unified.keys():
            dim += G_unified[key]
        return G_unified, dim, L_unified

    def init_state(self):
        self.population = np.random.rand(2 * self.num_pop, self.num_dim)
        self.skill_factor = np.array([i % self.num_task for i in range(2 * self.num_pop)])
        self.factorial_cost = np.full([2 * self.num_pop, self.num_task], np.inf)
        self.scalar_fitness = np.empty([2 * self.num_pop])
    
    def find_scalar_fitness(self):
        return 1 / np.min(np.argsort(np.argsort(self.factorial_cost, axis = 0), axis = 0) + 1, axis = 1)
    
    def sbx_crossover(self, p1, p2):
        D = p1.shape[0]
        cf = np.empty([D])
        u = np.random.rand(D)
        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (self.sbxdi + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), ( -1 / (self.sbxdi + 1)))
        child1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
        child1 = np.clip(child1, 0, 1)
        child2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
        child2 = np.clip(child2, 0, 1)
        return child1, child2
        

    def mutate(self, p):
        mp = float(1.0 / p.shape[0])
        u = np.random.uniform(size = [p.shape[0]])
        r = np.random.uniform(size = [p.shape[0]])
        tmp = np.copy(p)
        for i in range(p.shape[0]):
            if r[i] < mp:
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** ( 1.0 / (1 + self.pmdi)) - 1
                    tmp[i] = p[i] + delta * p[i]
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1.0 / (1 + self.pmdi))
        return tmp 
        
    def update(self):
        for i in range(2 * self.num_pop):
            sf = self.skill_factor[i]
            self.factorial_cost[i,sf] = self.tasks[sf].fitness(self.population[i], self.G_unified)
        self.scalar_fitness = self.find_scalar_fitness()
        
        sort_index = np.argsort(self.scalar_fitness[: : -1])
        self.population = self.population[sort_index]
        self.skill_factor = self.skill_factor[sort_index]
        self.factorial_cost = self.factorial_cost[sort_index]
        self.factorial_cost[self.num_pop:, :] = np.inf
        
        
    def run(self):
        self.init_state()
        self.update()
        for gen in range(self.num_gen):
            permutation = np.random.permutation(self.num_pop)
            self.population[: self.num_pop] = self.population[: self.num_pop][permutation]
            self.skill_factor[: self.num_pop] = self.skill_factor[: self.num_pop][permutation]
            self.factorial_cost[:self.num_pop] = self.factorial_cost[:self.num_pop][permutation]
            
            if self.rmp == 0:
                single_task_index = []
                for k in range(self.num_task):
                    single_task_index += list(np.where(self.skill_factor[: self.num_pop] == k)[0])
                self.population[: self.num_pop] = self.population[: self.num_pop][single_task_index]
                self.skill_factor[: self.num_pop] = self.skill_factor[: self.num_pop][single_task_index]
                self.factorial_cost[: self.num_pop] = self.factorial_cost[: self.num_pop][single_task_index]
                
            for i in range(0, self.num_pop, 2):
                p1, p2 = self.population[i], self.population[i + 1]
                sf1, sf2 = self.skill_factor[i], self.skill_factor[i + 1]
                
                if sf1 == sf2:
                    child1, child2 = self.sbx_crossover(p1, p2)
                    self.skill_factor[self.num_pop + i] = sf1
                    self.skill_factor[self.num_pop + i + 1] = sf1
                    
                elif np.random.rand() < self.rmp:
                    child1, child2 = self.sbx_crossover(p1, p2)
                    
                    if np.random.rand() < 0.5:
                        self.skill_factor[self.num_pop + i] = sf1
                    else:
                        self.skill_factor[self.num_pop + i] = sf2 
                    
                    if np.random.rand() < 0.5:
                        self.skill_factor[self.num_pop + i + 1] = sf1 
                    else:
                        self.skill_factor[self.num_pop + i + 1] = sf2 
                else:
                    child1 = np.copy(p1)
                    child2 = np.copy(p2)
                    self.skill_factor[self.num_pop + i] = sf1 
                    self.skill_factor[self.num_pop + i + 1] = sf2 
                    
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                sf1 = self.skill_factor[self.num_pop + i]
                sf2 = self.skill_factor[self.num_pop + i + 1]
                
            self.update()
            best_fitness = np.min(self.factorial_cost, axis = 0)
            mean_fitness = [np.mean(self.factorial_cost[:, i][np.isfinite(self.factorial_cost[:, i])]) for i in range(self.num_task)]
            info = ",".join([str(gen),
                            ','.join(['%f' %_ for _ in best_fitness]),
                            ','.join(['%f' %_ for _ in mean_fitness])])
            print('[INFO] %s' %info)
            # self.logger.info(info)
        
        
        