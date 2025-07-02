# Machine Learning Assisted-Genetic Algorithm
## Introduction
This project aims to provide a flexible automated solution for designing dual band (may be easily extended by altering the fitness calculation method) planar antennas. The whole optimization process is automated by utilizing Python scripts to control CST Studio Suite® 2025 (2023 is also valid). Genetic algorithm (BPSO is another possible method and may be adopted in the future) is used to optimize the antenna topology. The antenna topogy will gradually evolve from a arbitrary design to a spec-satisfying design. The design region of the antenna topology is a pixelated rectangular area with user-defined length, width, height, fineness, and position. This project can be easily integrated with any CST simulation files by directly drawing the design region at the user-defined location in the files. Machine learning-based surrogate model can be used to accelerate the genetic algortihm optimization process by roughly 10~15% faster in average (optional). 

---
## Setup
Before running the code, ensure you have the following installed:
- **CST Studio Suite®**: To solve EM problems. Official interface only support python 3.6~3.9.
- **Git**: To clone this repository.
- **Miniconda** or **Anaconda**: To manage the Python environment and dependencies.

### 1. Install Git
- **Windows**: Download and install Git from [git-scm.com](https://git-scm.com/downloads). Follow the default installation options.

Verify Git is installed by running:
```
git --version
```

### 2. Install Miniconda
Miniconda is a lightweight version of Anaconda that manages Python environments.
- Download Miniconda from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) (choose the version for your OS: Windows, Mac, or Linux).
- Follow the installer instructions:
  - **Windows**: Double-click the `.exe` file and follow the prompts. Check "Add Miniconda to my PATH" if available.
- Restart your terminal after installation.

Verify Miniconda is installed by running:
```
conda --version
```

### 3. Clone the Repository
In a terminal or command prompt, navigate to the folder where you want the project and run:
```
git clone https://github.com/electronics10/MLAO2.git
```
Then, enter the project directory:
```
cd MLAO2
```

### 4. Set Up the Conda Environment
All dependencies can be find in two of my other projects [[1]](https://github.com/electronics10/Topology_Optimization) and [[2]](https://github.com/electronics10/mlpifa).

After dependencies are installed:
- Activate the environment:
```
conda activate my_environment_name
```

### 5. Run the Code
1. Users can change the settings in `settings.json` for tailored use (e.g., S11 goal, design region size, hyperparameters).
2. Initialize the pixelated design region in a ".cst" file (automatically create `./CST_Antennas/topop.cst` if not specified). 
```
python initialize.py
```
3. Excitation port and other setup (e.g., GPU acceleration) in the ".cst" file should be defined by the user before optimization.
4. Design the antenna by running optimization code.
```
python GA.py
```
or
```
python MLAOGA.py
```
for ML usage. Random seed is assigned (2 in default) for replication in case CST crashes (S11 files would be saved to `./s11` for each antenna topology simulation). General optimization process info will be saved to `./data/log_seed.txt` and `./data/GA_seed.csv`. The final design will be plot after optimization.
5. Arbritray antenna topologies (solved in the process) and their corresponding S11 can be plot by running:
```
python plot.py
```
6. One can load arbritrary antenna topology into CST for later usage.
```
python verify.py
```


