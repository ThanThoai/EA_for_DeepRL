from abc import ABC, abstractmethod 
import numpy as np
from functools import wraps
from time import time 
from datetime import datetime 
import copy 
from typing import Callable, Tuple, List
import gym



class NeuralNetwork(ABC):
    @abstractmethod
    def get_weights_biases(self) -> np.array:
        pass 
    
    @abstractmethod
    def update_weights_biases(self, weights_bias: np.array) -> None:
        pass 
    
    def load(self, file):
        self.update_weights_biases(np.load(file))
        

class ActivationFunctions:
    @staticmethod
    def ReLu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def Sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def LeakyRele(x):
        return x if x > 0 else 0.01 * x
    

class MLP(NeuralNetwork):
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases_1 = np.random.randn(self.hidden_size)
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size)
        self.biases_2 = np.random.rand(self.output_size)
        self.ReLu = ActivationFunctions.ReLu
        self.Sigmoid = ActivationFunctions.Sigmoid
        
    
    def forward(self, x: np.array) -> np.array:
        x = x @ self.weights_1 + self.biases_1
        x = self.ReLu(x)
        x = x @ self.weights_2 + self.biases_2
        return self.Sigmoid(x)
    
    def get_weights_biases(self) -> np.array:
        w_1 = self.weights_1.flatten()
        w_2 = self.weights_2.flatten()
        print(np.concatenate((w_1, self.biases_1, w_2, self.biases_2), axis=0).shape)
        return np.concatenate((w_1, self.biases_1, w_2, self.biases_2), axis=0)

    def update_weights_biases(self, weights_biases: np.array) -> None:
        w_1, b_1, w_2, b_2 = np.split(weights_biases,
                                      [self.weights_1.size, self.weights_1.size + self.biases_1.size,
                                       self.weights_1.size + self.biases_1.size + self.weights_2.size])
        self.weights_1 = np.resize(w_1, (self.input_size, self.hidden_size))
        self.biases_1 = b_1
        self.weights_2 = np.resize(w_2, (self.hidden_size, self.output_size))
        self.biases_2 = b_2


class Individual(ABC):
    def __init__(self, input_size, hidden_size, output_size):
        self.nn = self.get_model(input_size, hidden_size, output_size)
        self.fitness = 0.0
        self.weights_biases: np.array = None
        
    def calculate_fitness(self, env) -> None:
        self.fitness, self.weights_biases = self.run_single(env)
        
    def update_model(self) -> None:
        self.nn.update_weights_biases(self.weights_biases)
        
    @abstractmethod
    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        pass 
    
    @abstractmethod
    def run_single(self, input_size, hidden_size, output_size) -> Tuple[float, np.array]:
        pass 
    
    
def crossover(parent1_weights_biases: np.array, parent2_weights_biases: np.array, p: float):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    if np.random.rand() < p:
        child1_weights_biases[position:], child2_weights_biases[position:] = \
            child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases


def crossover_new(parent1_weights_biases: np.array, parent2_weights_biases: np.array):
    position = np.random.randint(0, parent1_weights_biases.shape[0])
    child1_weights_biases = np.copy(parent1_weights_biases)
    child2_weights_biases = np.copy(parent2_weights_biases)

    child1_weights_biases[position:], child2_weights_biases[position:] = \
        child2_weights_biases[position:], child1_weights_biases[position:]
    return child1_weights_biases, child2_weights_biases


def inversion(child_weights_biases: np.array):
    return child_weights_biases[::-1]


def mutation_gen(child_weights_biases: np.array, p_mutation):
    for i in range(len(child_weights_biases)):
        if np.random.rand() < p_mutation:
            child_weights_biases[i] = np.random.uniform(-100, 100)


def mutation(parent_weights_biases: np.array, p: float):
    child_weight_biases = np.copy(parent_weights_biases)
    if np.random.rand() < p:
        position = np.random.randint(0, parent_weights_biases.shape[0])
        n = np.random.normal(np.mean(child_weight_biases), np.std(child_weight_biases))
        child_weight_biases[position] = n + np.random.randint(-10, 10)
    return child_weight_biases


def ranking_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    parent1, parent2 = sorted_population[:2]
    return parent1, parent2


def roulette_wheel_selection(population: List[Individual]):
    total_fitness = np.sum([individual.fitness for individual in population])
    selection_probabilities = [individual.fitness / total_fitness for individual in population]
    pick = np.random.choice(len(population), p=selection_probabilities)
    return population[pick]


def statistics(population: List[Individual]):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'Elapsed time:  {(end - start) :.3f}s')
        return result 
    return wrapper

class Population:
    def __init__(self, individual, pop_size, max_generation, p_mutation, p_crossover, p_inversion):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.p_mutation = p_mutation
        self.p_crossover = p_crossover
        self.p_inversion = p_inversion
        self.old_population = [individual for _ in range(pop_size)]
        self.new_population = []
        
    @timing 
    def run(self, env, run_generation: Callable, verbose=False, log=False, output_folder=None):
        for i in range(self.max_generation):
            [p.calculate_fitness(env) for p in self.old_population]
            self.new_population = [None for _ in range(self.pop_size)]
            run_generation(env,
                           self.old_population,
                           self.new_population,
                           self.p_mutation,
                           self.p_crossover,
                           self.p_inversion)
            if log:
                self.save_logs(i, output_folder)
            if verbose:
                self.show_stats(i)
            self.update_old_population()
        self.save_model_parameters(output_folder)
    
    def save_logs(self, n_gen, output_folder):
        date = self.now()
        file_name = 'logs.csv'
        mean, min, max = statistics(self.new_population)
        stats = f'{date},{n_gen},{mean},{min},{max}\n'
        with open(output_folder + file_name, 'a') as f:
            f.write(stats)
            
    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print(stats)

    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def save_model_parameters(self, output_folder):
        best_model = self.get_best_model_parameters()
        date = self.now()
        file_name = self.get_file_name(date) + '.npy'
        np.save(output_folder + file_name, best_model)

    def get_best_model_parameters(self) -> np.array:
        individual = sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]
        return individual.weights_biases      
    
    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}'.format(date,
                                                                              self.new_population[0].__class__.__name__,
                                                                              self.pop_size,
                                                                              self.max_generation,
                                                                              self.p_mutation,
                                                                              self.p_crossover)
    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')
    

class MLPIndividual(Individual):
    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return MLP(input_size, hidden_size, output_size)
    
    def run_single(self, env, n_episodes = 300, render = False) -> Tuple[float, np.array]:
        obs = env.reset()
        fitness = 0
        for _ in range(n_episodes):
            if render:
                env.render()
            action = self.nn.forward(obs)
            obs, reward, done, _ = env.step(round(action.item()))
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()

def generation(env, old_population, new_population, p_mutation, p_crossover, p_inversion=None):
    for i in range(0, len(old_population) - 1, 2):
        parent1 = roulette_wheel_selection(old_population)
        parent2 = roulette_wheel_selection(old_population)

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2
            
def test_model(nn, file):
    global observation 
    nn.load(file)
    score = 0
    done = False
    while(done == False):
        env.render()
        action = nn.forward(observation)
        observation, reward, done, info = env.step(round(action.item()))
        score += reward
        print(score)
    print(score)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    env.seed(123)
    train = True
    if train:
        POPULATION_SIZE = 100
        MAX_GENERATION = 100
        MUTATION_RATE = 0.4
        CROSSOVER_RATE = 0.9
        INPUT_SIZE = 4
        HIDDEN_SIZE = 2
        OUTPUT_SIZE = 1
        p = Population(MLPIndividual(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE),
                    POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, 0)
        p.run(env, generation, verbose=True, output_folder='./cartpole', log=True)
        env.close()
    else:
        env = gym.wrappers.Monitor(env, 'cartpole', video_callable=lambda episode_id: True, force=True)
        observation = env.reset()
        nn = MLP(4, 2, 1)
        test_model(nn, './cartpole04-11-2020_01-28_NN=MLPIndividual_POPSIZE=100_GEN=100_PMUTATION_0.4_PCROSSOVER_0.9.npy')
        env.close()
        