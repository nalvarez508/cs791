# Base File - AutoML/Traditional methods will both use this as starting point
# Time spent configuring this file is NOT counted towards total time for either methods
## Data imports would be the same for both.

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from copy import deepcopy

class NetworkTraffic:
  def __init__(self, filename=None, testSize=0.4, doNorm=False, doNormAll=False, doTransform=False):
    self.all_data = None
    self.testSize = testSize
    self.features = pd.read_csv("../ss_7_features.csv")
    self.feature_list = list(self.features.feature.values)

    # Import specific file
    if type(filename) == str:
      if doTransform:
        df = pd.read_csv(filename)
        df = df.sort_values(by=['label']).reset_index().drop(columns=["index"])
        df.fillna(df.groupby(['label'], as_index=False).mean(), inplace=True)

        for index, row in self.features.iterrows():
          df[row['feature']] = df[row['feature']] / df[row['normalizer']]

        df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        self.all_data = deepcopy(df)

      else: self.all_data = np.genfromtxt(filename, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
    
    # Import all files as single array 
    elif type(filename) == list:
      for file in filename:
        if file.endswith(".csv"):
          if doTransform:
            df = pd.read_csv(file)
            df = df.sort_values(by=['label']).reset_index().drop(columns=["index"])
            df.fillna(df.groupby(['label'], as_index=False).mean(), inplace=True)
            for index, row in self.features.iterrows():
              df[row['feature']] = df[row['feature']] / df[row['normalizer']]
            df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            try:
              self.all_data = pd.concat([self.all_data, df])
            except Exception as e:
              print(e)
              self.all_data = df
          else:
            try:
              self.all_data = np.vstack(np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26)))
            except:
              self.all_data = np.genfromtxt(file, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
      #print(self.all_data.shape)
    # Transfer_ID is excluded from this import
    
    if not doTransform: self.trimmed_all_data = self.turnInto2DArray()
    if doTransform:
      self.all_data = self.all_data[self.all_data.report_sec == 10]
      self.data = self.all_data[self.feature_list]

      #pd.to_numeric(self.data)
    else:
      self.data = np.delete(self.trimmed_all_data, 24, 1) # Remove labels
      self.data = np.delete(self.data, 0, 1) # Remove report_sec
      self.data = self.data.astype(float)
    if doTransform:
      self.target = self.all_data.label
      #pd.to_numeric(self.target)
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

class NT2:
  def __init__(self, filename=None, transform=False, drop=True):
    self.df = pd.DataFrame()
    if type(filename) == str:
      self.df = pd.read_csv(filename)
    elif type(filename) == list:
      for file in filename:
        if file.endswith(".csv"):
          thisFile = pd.read_csv(file)
          if self.df.empty: self.df = deepcopy(thisFile)
          else: self.df = pd.concat([self.df, thisFile], ignore_index=True)

    try: self.features = pd.read_csv('../ss_7_features.csv')
    except: self.features = pd.read_csv("ss_7_features.csv")
    self.feature_list = list(self.features.feature.values)
    self.setup(transform)
    self.data = self.df.drop(['label', 'report_sec', 'transfer_id'], axis=1)
    if drop: self.data = self.data[self.feature_list]
    self.target = preprocessing.LabelEncoder().fit_transform(self.df.label)
  
  def setup(self, t):
    self.df = self.df.sort_values(by=['label']).reset_index().drop(columns=["index"])
    self.df = self.df[self.df.label.isin(list(range(1,7)))] 
    self.df.fillna(self.df.groupby(['label'], as_index=False).mean(), inplace=True)

    if t:
      for index, row in self.features.iterrows():
        self.df[row['feature']] = self.df[row['feature']] / self.df[row['normalizer']]

    self.df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    self.df = self.df[self.df.report_sec == 10]

if __name__ == "__main__":
  import os
  os.chdir("data")
  NT2("b100d30.csv")
  NT2(["b100d30.csv", "b1000d30.csv"])