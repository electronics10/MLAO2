import numpy as np
import pandas as pd
import csv
import os
from settings import*
from GA import GA_Optimizer
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import torch.nn as nn
import torch.optim as optim


class AntennaCNN(nn.Module):
    def __init__(self):
        super(AntennaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, NX*NY, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten())
        flat_size = NX*NY * NX * NY
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, NX*NY),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(NX*NY, 1))

    def forward(self, x):
        x = x.view(-1, 1, NX, NY).float()
        return self.fc_layers(self.conv_layers(x))

class MLAO_GA(GA_Optimizer):
    def __init__(self):
        super().__init__()
        self.switch = 0
        self.model_mse = 100
        self.threshold = 0.2
        self.ml_folder = "artifacts_CNN"
        os.makedirs(f"./{self.ml_folder}", exist_ok=True)
        self.training_data_path = "artifacts_CNN/training_data.csv"
        if os.path.exists(self.training_data_path): os.remove(self.training_data_path)
        self.model = AntennaCNN()
    
    def predict(self, population):
        folder = self.ml_folder
        device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
        # Evaluation and plotting
        x_scaler = pickle.load(open(f"{folder}/x_scaler.pkl", "rb"))
        y_scaler = pickle.load(open(f"{folder}/y_scaler.pkl", "rb"))
        X_scaled = x_scaler.transform(population)
        X_test = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        model = self.model.to(device)
        model.load_state_dict(torch.load(f"{folder}/model.pth")) # Load best model saved
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

        scored_pop = []
        for i in range(len(population)):
            scored_pop.append((population[i], y_pred[i][0]))
        return scored_pop
    
    def create_training_data(self, scored_pop):
        with open(self.training_data_path, 'a', newline='') as csvfile:
            for item in scored_pop:
                individual = item[0]
                fitness = item[1]
                writer = csv.writer(csvfile)
                writer.writerow(list(np.ravel(individual)) + [fitness])
                writer.writerow(list(np.ravel(individual[::-1])) + [fitness]) # symmetry, double data size

    def train(self): 
        folder = self.ml_folder
        os.makedirs(folder, exist_ok=True)
        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
        # Load dataset
        data = pd.read_csv(f'{self.training_data_path}').values
        x_data = data[:, :NX*NY]
        y_data = data[:, NX*NY:]
        # Normalize inputs and outputs separately
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_data = x_scaler.fit_transform(x_data)
        y_data = y_scaler.fit_transform(y_data)
        # Split training set and validation set
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        # Save scalers
        pickle.dump(x_scaler, open(f"{folder}/x_scaler.pkl", "wb"))
        pickle.dump(y_scaler, open(f"{folder}/y_scaler.pkl", "wb"))
        
        # Model, Loss, Optimizer
        model = self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # Early stopping settings
        early_stopping_patience = 70
        best_loss = float('inf')
        patience_counter = 0
        # Training loop
        num_epochs = 5000
        loss_list = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                loss_list.append([epoch, loss.item(), test_loss.item()])
            # if epoch % 50 == 0: print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
            # Early stopping
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                patience_counter = 0
            else: patience_counter += 1
            if patience_counter >= early_stopping_patience: break

        # Save the model
        torch.save(model.state_dict(), f'{folder}/model.pth')
        mse = test_loss.item()
        # print("mse: ", mse)
        self.model_mse = mse

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
            ## Logic for using model or CST solver
            max_MLG = 20 # max consecutive ML generations
            avg_last = 0
            if self.switch == 0: self.population_cache = population # Bad prediction handle
            if self.model_mse < self.threshold and self.switch <= max_MLG: # Use CNN model
                if self.switch == max_MLG: # final consecutive ML generation round
                    scored_pop, pop_avg_fitness = self.fitness_assign_and_sort(population) # run CST for final check
                    best_fitness = scored_pop[0][1]
                    self.switch = 0 # CST used
                    self.create_training_data(scored_pop)
                    self.train() # CST used
                    ### Bad overall model usage
                    if pop_avg_fitness < avg_last or best_fitness < self.fitness_record[-1] - ML_ERROR_TOLERANCE:
                        print("model failed")
                        best_fitness = self.fitness_record[-1]
                        for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record previous fitness
                        scored_pop, pop_avg_fitness = self.fitness_assign_and_sort(self.population_cache)
                        for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
                        best_topology = self.encode_B2T(scored_pop[0][0])
                        print(f"Iterations: {len(self.fitness_record)}")
                        print(f"Best Fitness: {best_fitness} ({best_topology})\n")
                        self.switch = 0 # CST used
                        self.create_training_data(scored_pop)
                        self.train() # CST used
                    ### Model did enhance exploration ability locally
                    else:
                        for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
                        print("Model workeddddddddddddddddddddddddddd")
                        best_topology = self.encode_B2T(scored_pop[0][0])
                        print(f"Iterations: {len(self.fitness_record)}")
                        print(f"Best Fitness: {best_fitness} ({best_topology})\n")
                else: ## Use model prediction for max_MLG consecutive generations before final round
                    print("Predicting...")
                    scored_pop = self.predict(population)
                    scored_pop.sort(key=lambda x: x[1], reverse=True)
                    best_fitness = scored_pop[0][1]
                    best_topology = self.encode_B2T(scored_pop[0][0])
                    for i in range(POPULATION_SIZE): pop_avg_fitness += scored_pop[i][1]
                    pop_avg_fitness = pop_avg_fitness/POPULATION_SIZE
                    self.switch += 1
            else: # Use normal GA with CST since model error is too large
                scored_pop, pop_avg_fitness = self.fitness_assign_and_sort(population)# Use CST
                for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
                best_topology = self.encode_B2T(scored_pop[0][0])
                print(f"Iterations: {len(self.fitness_record)}")
                print(f"Best Fitness: {best_fitness} ({best_topology})\n")
                self.switch = 0 # CST used
                self.create_training_data(scored_pop)
                self.train() # CST used
            # ML criterion
            avg_last = pop_avg_fitness

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
            f.write(f"Iterations: {len(self.fitness_record)}\n")
            f.write(f"Pop avg Fitness: {pop_avg_fitness}\n")
            f.write(f"Best Fitness: {best_fitness} ({best_topology})\n")
            f.close()
            generation += 1

            # Record progression
            df = pd.DataFrame(self.fitness_record, columns = ['best_fitness'])
            df.to_csv(f'data/progression_{SEED}.csv', index=False)

        print("Converged.\n")
        import plot as myplt
        myplt.plot_s11([best_topology])
        myplt.plot_topology([best_topology])
        myplt.plt.show()

if __name__ == "__main__":
    optimizer = MLAO_GA()
    print(f"SEED = {SEED}")
    optimizer.evolve()