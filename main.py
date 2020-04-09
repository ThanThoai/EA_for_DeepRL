from task import CartPole
from MFEA import MFEA

SBXDI = 2 
PMDI  = 5
NUM_POP = 20
NUM_GEN = 100



if __name__ == '__main__':
    
    tasks = [CartPole()]
    for id in range(20):
        MFEA(tasks, NUM_POP, NUM_GEN, SBXDI, PMDI, 0).run()
        MFEA(tasks, NUM_POP, NUM_GEN, SBXDI, PMDI, 1).run()
