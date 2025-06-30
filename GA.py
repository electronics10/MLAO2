import numpy as np
import pandas as pd
import csv
import os
from settings import*
import CST_Controller as cstc

class Optimizer:
    def __init__(self):
        self.antenna = cstc.CSTInterface(FILEPATH)

    def acquire_S11_from_CST(self, binary_array):
        self.antenna.delete_results() # delete legacy
        self.antenna.update_distribution(binary_array)
        self.antenna.start_simulate()
        s11 = self.antenna.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        index = int(''.join(map(str, binary_array)), 2) # binary back to decimal
        data.to_csv(f's11/s11_{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 's11/s11_{index}.csv'")

    def calculate_fitness(self, binary_array, target=(FREQ_L,FREQ_H), goal=GOAL):
        index = int(''.join(map(str, binary_array)), 2) # binary back to decimal
        # Check fitness table
        try: fitness_table = pd.read_csv("fitness_table.csv", header=None, sep=' ')
        except: fitness_table = pd.DataFrame([[-100,-100]]) # fake table if not exist
        condition = fitness_table.iloc[:,0] == f'{index}' # fitness calculated before
        if condition.any(): fitness = fitness_table.loc[condition, 1].iloc[0]
        else: 
            self.acquire_S11_from_CST(binary_array)
            # Calculate Fitness
            lfreq = target[0]
            hfreq = target[1]
            df = pd.read_csv(f"./s11/s11_{index}.csv")
            j = 0
            n = 0
            freq = 0
            fitness = 0
            while freq < hfreq:
                freq = df.iloc[j, 0] # Read fequency
                if freq >= lfreq:
                    s11 = df.iloc[j, 1] # Read s11
                    # fitness += (goal - s11) # Record fitness # the larger the merrier
                    fitness += max(goal, s11) 
                    n += 1
                j += 1
            fitness = fitness/n/goal
            # Save to fitness_table
            with open("fitness_table.csv", 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ')
                spamwriter.writerow([index, fitness])
        
        return fitness

    def fitness_assign_and_sort(self, population):
        scored_pop = []
        pop_avg_fitness = 0
        for i in range(len(population)):
            individual = population[i]
            fitness = self.calculate_fitness(individual)
            scored_pop.append((individual, fitness))
            pop_avg_fitness += fitness
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        pop_avg_fitness = pop_avg_fitness/len(population)
        return scored_pop, pop_avg_fitness

    def crossover(self, p1, p2):
        mask = np.random.randint(0, 2, size=p1.shape)
        return np.where(mask, p1, p2)

    def mutate(self, individual, rate=0.1):
        mask = np.random.rand(*individual.shape) < rate
        return np.logical_xor(individual, mask).astype(int)

    def run(self, store = False):
        # Create population
        np.random.seed(SEED)
        pixels = NX*NY
        pop_indices = np.random.randint(pixels, size = POPULATION_SIZE)
        population = []
        for i in range(POPULATION_SIZE):
            individual = np.array([int(bit) for bit in format(pop_indices[i], f'0{pixels}b')])
            population.append(individual)
        population = np.array(population)
        
        # Evolve till spec satisfied
        generation = 0
        best_fitness = -1000 # negative infinity
        fitness_record = []
        while (best_fitness < CONVERGENCE) or (len(fitness_record) < MAX_ITERATIONS):
            # Fitness Assignment
            scored_pop, pop_avg_fitness = self.fitness_assign_and_sort(population) # [(individual, fitness)]
            best_fitness = scored_pop[0][1]
            best_index = int(''.join(map(str, scored_pop[0][0])), 2) # binary to decimal
            for i in range(POPULATION_SIZE): fitness_record.append(best_fitness)

            # Selection
            elites_size = int(POPULATION_SIZE*SELECT_RATE)
            elites = scored_pop[:elites_size] # [(individual, fitness)]

            new_pop = [] # new population
            for item in elites: new_pop.append(item[0]) # append elite individuals

            # Crossover and Mutation
            while len(new_pop) < POPULATION_SIZE:
                # Randomly select two parents from elites
                p_indices = np.random.randint(elites_size, size=2)
                p1 = np.array(elites[p_indices[0]][0])
                p2 = np.array(elites[p_indices[1]][0])
                child = self.mutate(self.crossover(p1, p2), rate = MUTATE_RATE)
                new_pop.append(child)
            population = np.array(new_pop)

            print(f"\nGen{generation}")
            print(f"Iterations: {(generation)*POPULATION_SIZE}")
            print(f"Pop avg Fitness: {pop_avg_fitness}")
            print(f"Best Fitness: {best_fitness} ({best_index})")
            path = 'output.txt'
            f = open(path, 'a')
            f.write(f"\nGen{generation}")
            f.write(f"Iterations: {(generation)*POPULATION_SIZE}")
            f.write(f"Pop avg Fitness: {pop_avg_fitness}")
            f.write(f"Best Fitness: {best_fitness} ({best_index})")
            f.close()
            generation += 1

        if store:
            os.makedirs("./data", exist_ok=True)
            df = pd.DataFrame(fitness_record, columns = ['best_fitness'])
            df.to_csv(f'data/GA_{SEED}.csv', index=False) 

if __name__ == "__main__":
    os.makedirs("./s11", exist_ok=True) # create folder "s11"
    optimizer = Optimizer()
    optimizer.run(store = True)
