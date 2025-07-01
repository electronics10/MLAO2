# Machine Learning Assisted-Genetic Algorithm
## Introduction
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
