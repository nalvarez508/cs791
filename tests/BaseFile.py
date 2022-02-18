import sys
sys.path.append("../../")

import modules
import numpy as np
from sklearn import model_selection

mydata = modules.NT("../../data/b100d30.csv")
x_train, x_test, y_train, y_test = model_selection.train_test_split(mydata.data, mydata.target, test_size=0.4, random_state=508)