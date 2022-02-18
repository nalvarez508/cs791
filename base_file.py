# Base File - AutoML/Traditional methods will both use this as starting point
# Time spent configuring this file is NOT counted towards total time for either methods
## Data imports would be the same for both.
import numpy as np
import modules

data = np.genfromtxt
myData = modules.NT("data/b100d30.csv")