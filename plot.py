import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import settings

def encode_B2T(binary_array):
        decimal_value = int(''.join(map(str, binary_array)), 2)
        hex_str = hex(decimal_value)
        return hex_str
    
def decode_T2B(topology):
    return np.array([int(bit) for bit in format(int(topology, 16), f'0{NX*NY}b')])

def plot_s11(indices):
    plt.figure("S11")
    for i in indices:
        df = pd.read_csv(f"./s11/s11_{i}.csv")  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label=f'{i}')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("|S11| (dB)")
        plt.title("S11")
        plt.legend(loc = 'lower left')
        # plt.grid()
    plt.savefig(f"figures/s11_{i}.png")
    
def plot_topology(indices):
    for i in indices:
        plt.figure(i)
        binar_array = decode_T2B(i)
        plt.imshow(binar_array.reshape(NX, NY), cmap='binary', vmin=0, vmax=1, origin = 'lower')

        # Turn off ticks and labels
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Topology {i}")
        plt.tight_layout()
        plt.xlabel(f'{WW} (mm)')
        plt.ylabel(f'{LL} (mm)')

        plt.savefig(f"figures/topology_{i}.png")


if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    indices = TOPOLOGIES.split()
    plot_s11(indices)
    plot_topology(indices)
