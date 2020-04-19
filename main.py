from task import CartPole
from MFEA import MFEA

SBXDI = 2 
PMDI  = 5
NUM_POP = 20
NUM_GEN = 100



if __name__ == '__main__':
    
    tasks = [CartPole(shape_gen= [4, 3, 2, 1]),
             CartPole(shape_gen= [4, 2, 2, 1]),
             CartPole(shape_gen= [4, 2, 3, 4, 1])]
    for id in range(20):
        # MFEA(tasks, NUM_POP, NUM_GEN, SBXDI, PMDI, 0).run()
        MFEA(tasks, NUM_POP, NUM_GEN, SBXDI, PMDI, 1).run()
