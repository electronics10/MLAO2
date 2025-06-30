import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from settings import*

def read_best_topology(seed=SEED):
    pass

def plot_s11(topology):
    df = pd.read_csv(f"./s11/s11_{i}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label=f'{i}')
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S11| (dB)")
    plt.title("S11")
    plt.legend(loc = 'lower left')
        # plt.grid()
    
def plot_topology(indices):
    for i in indices:
        plt.figure(i)
        topology = [int(bit) for bit in format(i, '012b')]
        topology.insert(5,1)
        topology.insert(6,1)
        topology.insert(9,1)
        topology.insert(10,1)
        topology = np.array(topology).reshape((4,4))
        plt.imshow(topology, cmap='binary', vmin=0, vmax=1, origin = 'lower')

        # Turn off ticks and labels
        plt.xticks([])
        plt.yticks([])

        # Draw custom grid for 4x4 cells
        for x in range(5):  # 0 to 4
            plt.axhline(x - 0.5, color='gray', linewidth=0.5)
            plt.axvline(x - 0.5, color='gray', linewidth=0.5)

        plt.title(f"Topology {i}")
        plt.tight_layout()


if __name__ == "__main__":
    indices = list(map(int, input("index: ").split()))
    # indices = np.random.randint(4096, size = 300)
    plot_s11(indices)
    plot_topology(indices)
    plt.show()