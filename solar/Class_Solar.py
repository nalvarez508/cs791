import numpy as np
from datetime import datetime
from sklearn import preprocessing

class Solar:
  def __init__(self, datafile, skip_f=0):
    # Function to grab timestamp and IGNORE the year because that is insignificant
    dateconvert = lambda x: datetime.strptime(x.decode('ascii'), '%Y%m%d %H:%M').replace(year=1900)
    self.allArray = np.genfromtxt(datafile, delimiter=',', names=True, autostrip=True, converters={1: dateconvert}, dtype=(float, 'datetime64[m]', float, float, float, float, float, float, float, float, float, float, float, float, float), skip_footer=skip_f) #skip_footer=10942
    try:
      self.zone = self.allArray[u'\ufeffZONEID']#[:, 0]
    except ValueError:
      try:
        self.zone = self.allArray['ZONEID']
      except ValueError:
        self.zone = self.allArray[self.allArray.dtype.names[0]]
    self.timestamp = self.allArray['TIMESTAMP']#[:, 1]
    self.var78 = self.allArray['VAR78']#[:, 2]
    self.var79 = self.allArray['VAR79']#[:, 3]
    self.var134 = self.allArray['VAR134']#[:, 4]
    self.var157 = self.allArray['VAR157']#[:, 5]
    self.var164 = self.allArray['VAR164']#[:, 6]
    self.var165 = self.allArray['VAR165']#[:, 7]
    self.var166 = self.allArray['VAR166']#[:, 8]
    self.var167 = self.allArray['VAR167']#[:, 9]
    self.var169 = self.allArray['VAR169']#[:, 10]
    self.var175 = self.allArray['VAR175']#[:, 11]
    self.var178 = self.allArray['VAR178']#[:, 12]
    self.var228 = self.allArray['VAR228']#:, 13]
    self.power = self.allArray['POWER']#[:, 14]
    self.notnormdata = self.combineData()
    self.data = self.normalize(self.notnormdata)
    self.zonedata = self.normalize(np.split(self.notnormdata, np.unique(self.notnormdata[:, 0], return_index=True)[1][1:]))
    self.zonepower = np.split(self.power, np.unique(self.zone, return_index=True)[1][1:])

  def _timestamp(self):
    for i in range(0,25):
      print(self.timestamp[i])
    for i in range(len(self.timestamp)-25, len(self.timestamp)):
      print(self.timestamp[i])

  def normalize(self, a):
    if type(a) == list:
      result = list()
      for array in a:
        sclr = preprocessing.MinMaxScaler()
        sclr.fit(array)
        result.append(sclr.transform(array))
      return result
    elif type(a) == np.ndarray:
      sclr = preprocessing.MinMaxScaler()
      sclr.fit(a)
      return sclr.transform(a)

  def combineData(self):
    #dataDtype = np.dtype([('ZONEID', np.int64), ('TIMESTAMP', np.dtype('datetime65[m]')), ('VAR78')])

    attrList = self.getAttributes()[:-1]
    #print(attrList)
    dataArray = np.vstack((attrList[0]))
    for attr in attrList[1:]:
      dataArray = np.hstack((dataArray, np.vstack((attr.astype(np.float64)))))
    return dataArray
  
  def printAll(self):
    print(self.zone, self.timestamp, self.var78, self.var79, self.var134, self.var157, self.var164, self.var165, self.var166, self.var167, self.var169, self.var175, self.var178, self.var228, self.power)
  
  def getAttributes(self):
    return [self.zone, self.timestamp, self.var78, self.var79, self.var134, self.var157, self.var164, self.var165, self.var166, self.var167, self.var169, self.var175, self.var178, self.var228, self.power]
  
  def getNames(self):
    return self.allArray.dtype.names

  def correlation(self):
    #print(np.corrcoef(self.allArray[:, 2:15], rowvar=False))
    #print(np.corrcoef(self.var78, self.power))
    #print(self.getAttributes())
    tempArray = np.empty([0, self.allArray.shape[0]])
    for item in self.getAttributes():
      if str(item.dtype) == 'float64':
        tempArray = np.vstack((tempArray, item))
    np.set_printoptions(precision=5, linewidth=151, suppress=True)
    #print("Correlation of an attribute to other attributes")
    return (np.corrcoef(tempArray))