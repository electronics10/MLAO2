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
cd Topology_Optimization
```

### 4. Set Up the Conda Environment
This project uses a predefined environment file (`environment.yml`) to install all dependencies.

- Create the environment (for Windows only):
```
conda env create -f environment.yml
```
- Activate the environment:
```
conda activate autotune
```

### 5. Run the Code
- Run the main script:
```
python main.py
```


### Troubleshooting
- If `conda` commands don’t work, ensure Miniconda is added to your system PATH or restart your terminal.
- For errors during `conda env create`, ensure you have an active internet connection, as it downloads packages.
- For errors about CST, ensure CST is added to your system PATH. It can also be done by altering the first few line in `Antenna_Design.py`, such as `sys.path.append(r"C:\Program Files (x86)\CST STUDIO SUITE 2023\AMD64\python_cst_libraries")`.
- Other CST errors might occur due to version difference and cause filepath name variation. One may check the correct file name in CST and change it in `Antenna_Design.py` according to prompt.
- Contact me for any further issues.
