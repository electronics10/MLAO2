import numpy as np
import pandas as pd
import csv
import os
from settings import*
import CST_Controller as cstc

class GA_Optimizer:
    def __init__(self):
        self.renew_fitness_table()
        os.makedirs("./s11", exist_ok=True) # create folder "s11"
        os.makedirs("./data", exist_ok=True) # create folder for fitness progression storage
        self.log_path = f'data/log_{SEED}.txt'
        f = open(self.log_path, 'w')
        f.write(f"SEED = {SEED}\n")
        f.close()
        self.antenna = cstc.CSTInterface(FILEPATH)
        self.fitness_record = []

    def encode_B2T(self, binary_array):
        decimal_value = int(''.join(map(str, binary_array)), 2)
        hex_str = hex(decimal_value)
        return hex_str
    
    def decode_T2B(self, topology):
        return np.array([int(bit) for bit in format(int(topology, 16), f'0{NX*NY}b')])
    
    def renew_fitness_table(self):
        with open("fitness_table.csv", 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ')
            directory = 's11'  # set directory path   
            for entry in os.scandir(directory):  
                # if entry.is_file():  # check if it's a file
                fitness = self.calculate_fitness(entry.path)
                topology = entry.path.split('s11_')[1].split('.csv')[0]
                spamwriter.writerow([topology, fitness])
                print(f"Fitness of topology {topology} calculated.")
        print("Fitness table renewed.\n")

    def acquire_S11_from_CST(self, binary_array):
        topology = self.encode_B2T(binary_array) # binary back to decimal
        print(f"Solving S11 for topology {topology}")
        self.antenna.delete_results() # delete legacy
        self.antenna.update_distribution(binary_array)
        self.antenna.start_simulate()
        s11 = self.antenna.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        data.to_csv(f's11/s11_{topology}.csv', index=False) # save to CSV
        print(f"S11 saved to 's11/s11_{topology}.csv'")

    def calculate_fitness(self, s11_path, target=(FREQ_L,FREQ_H), goal=GOAL):
        lfreq = target[0]
        hfreq = target[1]
        df = pd.read_csv(s11_path)
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
        return fitness

    def assign_fitness(self, binary_array):
        topology = self.encode_B2T(binary_array) # binary back to decimal
        # Check fitness table
        try: fitness_table = pd.read_csv("fitness_table.csv", header=None, sep=' ')
        except: fitness_table = pd.DataFrame([[-100,-100]]) # fake table if not exist
        condition = fitness_table.iloc[:,0] == topology # fitness calculated before
        if condition.any():
            print(f"Skip repeated calculation for topology {topology}")
            fitness = fitness_table.loc[condition, 1].iloc[0]
        else:
            self.acquire_S11_from_CST(binary_array)
            fitness = self.calculate_fitness(f's11/s11_{topology}.csv')
            # Save to fitness_table
            with open("fitness_table.csv", 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ')
                spamwriter.writerow([topology, fitness])
        return fitness

    def fitness_assign_and_sort(self, population):
        scored_pop = []
        pop_avg_fitness = 0
        for i in range(len(population)):
            individual = population[i]
            fitness = self.assign_fitness(individual)
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

    def evolve(self):
        # Create population
        np.random.seed(SEED)
        pixels = NX*NY
        pop_indices = np.random.randint(pixels, size = POPULATION_SIZE)
        population = []
        for i in range(POPULATION_SIZE):
            individual = hex(pop_indices[i])
            individual = self.decode_T2B(individual)
            population.append(individual)
        population = np.array(population)
        
        # Evolve till spec satisfied
        generation = 0
        best_fitness = -1000 # negative infinity
        while (best_fitness < CONVERGENCE) or (len(self.fitness_record) < MAX_ITERATIONS):
            print(f"\nGen{generation}")
            # Fitness Assignment
            scored_pop, pop_avg_fitness = self.fitness_assign_and_sort(population) # [(individual, fitness)]
            print(f"Iterations: {len(self.fitness_record)}")
            best_fitness = scored_pop[0][1]
            best_topology = self.encode_B2T(scored_pop[0][0])
            for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness)

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

            print(f"Pop avg Fitness: {pop_avg_fitness}")
            print(f"Best Fitness: {best_fitness} ({best_topology})")
            f = open(self.log_path, 'a')
            f.write(f"\nGen{generation}\n")
            f.write(f"Iterations: {len(self.fitness_record)}")
            f.write(f"Pop avg Fitness: {pop_avg_fitness}\n")
            f.write(f"Best Fitness: {best_fitness} ({best_topology})\n")
            f.close()
            generation += 1

            # Record progression
            df = pd.DataFrame(self.fitness_record, columns = ['best_fitness'])
            df.to_csv(f'data/progression_{SEED}.csv', index=False)

if __name__ == "__main__":
    optimizer = GA_Optimizer()
    print(f"SEED = {SEED}")
    optimizer.evolve()
