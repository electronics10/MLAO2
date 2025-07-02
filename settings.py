import json

def load_settings(config_path='settings.json'):
    with open(config_path, 'r') as f:
        return json.load(f)
    
try: settings = load_settings()
except Exception as e: print(f"Error loading settings.json: {e}")

# CST file path============================================
FILEPATH = settings['FILEPATH']

# Parameters of the Rectangular Design Region==============
LL = settings['LL'] # length (x-direction) (mm)
WW = settings['WW'] # width (y-direction) (mm)
HC = settings['HC'] # copper height (z-direction) (mm)
DD = settings['DD'] #  side length of each square pixel (mm)
NX= int(LL//DD)
NY = int(WW//DD)
## Absolute location of the lower-left-bottom corner of the rectangular design region
XX = settings['XX'] # x axis (mm)
YY = settings['YY'] # y axis (mm)
ZZ = settings['ZZ'] # z axis (mm)


# Optimization settings (Genetic Algorithm)
SEED = settings['SEED']

MAX_ITERATIONS = settings['MAX_ITERATIONS']
CONVERGENCE = settings['CONVERGENCE'] # fitness value (normalized to [0,1])
POPULATION_SIZE = int(CONVERGENCE*30)
SELECT_RATE = POPULATION_SIZE**(1/2)/POPULATION_SIZE
MUTATE_RATE = 1/POPULATION_SIZE*CONVERGENCE*30

ML_ERROR_TOLERANCE = 0.2*CONVERGENCE

# Goal
GOAL = settings['GOAL'] # S11 goal (dB)
LB_FREQ_L = settings['LB_FREQ_L'] #  lower frequency bound for fitness calculation 
LB_FREQ_H = settings['LB_FREQ_H'] #  upper frequency bound for fitness calculation 
HB_FREQ_L = settings['HB_FREQ_L'] #  lower frequency bound for fitness calculation 
HB_FREQ_H = settings['HB_FREQ_H'] #  upper frequency bound for fitness calculation 

# Frequency Range
FMIN = settings['FMIN'] # GHz
FMAX = settings['FMAX'] # GHz

# Plot topologies and S11
TOPOLOGIES = settings['TOPOLOGIES']

# Load topologgy into CST
VERIFICATION = settings['VERIFICATION']
