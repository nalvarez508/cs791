# Base File - AutoML/Traditional methods will both use this as starting point
# Time spent configuring this file is NOT counted towards total time for either methods
## Data imports would be the same for both.

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

class NetworkTraffic:
  def __init__(self, filename=None, testSize=0.4, doNorm=False, doNormAll=False, doTransform=False):
    self.all_data = None
    self.testSize = testSize
    self.features = features = pd.read_csv("../../ss_7_features.csv")
    self.feature_list = list(features.feature.values)

    # Import specific file
    if type(filename) == str:
      if doTransform:
        df = pd.read_csv(filename)
        df = df.sort_values(by=['label']).reset_index().drop(columns=["index"])

        for index, row in self.features.iterrows():
          df[row['feature']] = df[row['feature']] / df[row['normalizer']]
        self.all_data = df

      else: self.all_data = np.genfromtxt(filename, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
    
    # Import all files as single array 
    elif type(filename) == list:
      for file in filename:
        if file.endswith(".csv"):
          try:
            self.all_data = np.vstack(np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26)))
          except:
            self.all_data = np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
      #print(self.all_data.shape)
    # Transfer_ID is excluded from this import
    
    if not doTransform: self.trimmed_all_data = self.turnInto2DArray()
    if doTransform:
      self.allData = self.allData[self.allData.report_sec == 10]
      self.data = self.allData[self.feature_list]
      pd.to_numeric(self.data)
    else:
      self.data = np.delete(self.trimmed_all_data, 24, 1) # Remove labels
      self.data = np.delete(self.data, 0, 1) # Remove report_sec
      self.data = self.data.astype(float)
    if doTransform:
      self.target = self.allData.label
      pd.to_numeric(self.target)
    else:
      self.target = self.trimmed_all_data[:,-1]
      self.target = self.target.astype(int)

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
  def pick10thReportSec(self, d, doTransform):
    return d[np.where(d[:,0] == 10)]

  # Make all_data a 2D array, instead of 1D tuples
  def turnInto2DArray(self, doTransform=False):
    attrList = list()
    for attr in self.all_data.dtype.names:
      attrList.append(attr)
    dataArray = np.vstack(self.all_data[attrList[0]])
    # For each column, add it to DataArray
    for attr in attrList[1:]:
      dataArray = np.hstack((dataArray, np.vstack((self.all_data[attr]))))
    return self.pick10thReportSec(dataArray, doTransform)
  
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