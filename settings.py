# CST file path============================================
FILEPATH = "CST_Antennas/topop.cst"


# Parameters of the Rectangular Design Region==============
LL = 20 # length (x-direction) (mm)
WW = 10 # width (y-direction) (mm)
HC = 0.2 # copper height (z-direction) (mm)
DD = 5 #  side length of each square pixel (mm)
NX= int(LL//DD)
NY = int(WW//DD)
## Absolute location of the lower-left-bottom corner of the rectangular design region
XX = -10 # x axis (mm)
YY = -5 # y axis (mm)
ZZ = 0 # z axis (mm)


# Optimization settings (Genetic Algorithm)
SEED = 2

MAX_ITERATIONS = 2000
CONVERGENCE = 0.3 # fitness value
POPULATION_SIZE = int(CONVERGENCE*50)
SELECT_RATE = POPULATION_SIZE**(1/2)/POPULATION_SIZE
MUTATE_RATE = 1/POPULATION_SIZE*CONVERGENCE*30

ML_ERROR_TOLERANCE = 0.2*CONVERGENCE

# S11 Goal
GOAL = -6 # S11 goal (dB)
FREQ_L = 2.4 #  lower frequency bound for fitness calculation 
FREQ_H = 2.5 #  upper frequency bound for fitness calculation 
