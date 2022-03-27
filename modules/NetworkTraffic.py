# Base File - AutoML/Traditional methods will both use this as starting point
# Time spent configuring this file is NOT counted towards total time for either methods
## Data imports would be the same for both.

import numpy as np
from sklearn import preprocessing, model_selection

class NetworkTraffic:
  def __init__(self, filename=None, testSize=0.4, doNorm=False, doNormAll=False):
    self.all_data = None
    self.testSize = testSize

    # Import specific file
    if filename != None:
      self.all_data = np.genfromtxt(filename, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
    
    # Import all files as single array 
    else:
      from os import listdir, chdir
      chdir("../../../data")
      for file in listdir():
        if file.endswith(".csv"):
          try:
            self.all_data = np.vstack(np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26)))
          except:
            self.all_data = np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
      print(self.all_data.shape)
    # Transfer_ID is excluded from this import
    
    self.trimmed_all_data = self.turnInto2DArray()
    self.data = np.delete(self.trimmed_all_data, 24, 1) # Remove labels
    self.data = np.delete(self.data, 0, 1) # Remove report_sec
    self.target = self.trimmed_all_data[:,-1]

    if doNorm == True:
      if doNormAll == True:
        self.normalize(True)
      else:
        self.x_train, self.x_test, self.y_train, self.y_test = self.normalize()

  def createSets(self):
    x_train, x_val, y_train, y_val = model_selection.train_test_split(self.data, self.target, test_size=self.testSize, random_state=508)
    x_val, x_test, y_val, y_test = model_selection.train_test_split(x_val, y_val, test_size=0.45, random_state=508)
    return [x_train, x_val, x_test, y_train, y_val, y_test]

  # Only need one time slice from packet info
  def pick10thReportSec(self, d):
    return d[np.where(d[:,0] == 10)]

  # Make all_data a 2D array, instead of 1D tuples
  def turnInto2DArray(self):
    attrList = list()
    for attr in self.all_data.dtype.names:
      attrList.append(attr)
    dataArray = np.vstack(self.all_data[attrList[0]])
    # For each column, add it to DataArray
    for attr in attrList[1:]:
      dataArray = np.hstack((dataArray, np.vstack((self.all_data[attr]))))
    return self.pick10thReportSec(dataArray)
  
  # Normalize the data with a standard scaler if enabled
  def normalize(self, all=False):
    # Apply scaler to all data before splitting
    if all:
      scaler = preprocessing.MinMaxScaler().fit(self.data)
      self.data = scaler.transform(self.data)
    
    # Apply scaler to train and test data separately
    else:
      xtr, xte, ytr, yte = model_selection.train_test_split(self.data, self.target, test_size=self.testSize, random_state=508)
      scaler = preprocessing.MinMaxScaler().fit(xtr)
      xtr = scaler.transform(xtr)
      xte = scaler.transform(xte)
      return [xtr, xte, ytr, yte]

# Normalize coding time 24:39