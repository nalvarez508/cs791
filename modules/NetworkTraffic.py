# Base File - AutoML/Traditional methods will both use this as starting point
# Time spent configuring this file is NOT counted towards total time for either methods
## Data imports would be the same for both.

import numpy as np

class NetworkTraffic:
  def __init__(self, filename):
    self.all_data = np.genfromtxt(filename, dtype=None, delimiter=",", names=True, excludelist=["transfer_id_"], autostrip=True, usecols=range(1,26))
    # Transfer_ID is excluded from this import

    self.trimmed_all_data = self.turnInto2DArray()
    self.data = np.delete(self.trimmed_all_data, 24, 1)
    self.target = self.trimmed_all_data[:,-1]
    #print(self.data)
    #print(self.target)
  
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
  
