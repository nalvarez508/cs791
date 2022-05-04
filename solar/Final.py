import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from Class_Solar import Solar

NUMBER_ATTRIBUTES = 15
BEGIN_AT = 0
_VALUETOTEST = 8
_NUMBERZONES = 3
_ZONETOTEST = 2 # ZoneID - 1
PleaseShowMe = False
np.set_printoptions(precision=5, suppress=True)

s_train = Solar("solar_training.csv", skip_f=24)
print("Shape of training data (baseline):", s_train.data.shape)
s_train_24ahead = Solar("solar_training_24ahead.csv")
print("Shape of training data (24 hours ahead):", s_train_24ahead.data.shape)
s_test = Solar("solar_test.csv")
print("Shape of test data (baseline):", s_test.data.shape)
s_test_24behind = Solar("solar_test_24behind.csv", skip_f=24)
print("Shape of test data (24 hours behind):", s_test_24behind.data.shape)

RMSE_Scores = [0,0,0]
MAE_Scores = [0,0,0]

def truncate(num, digits):
  l = str(float(num)).split('.')
  digits = min(len(l[1]), digits)
  return l[0] + '.' + l[1][:digits]

def generatePlots():
  fig, axs = plt.subplots(NUMBER_ATTRIBUTES, NUMBER_ATTRIBUTES)
  fig.tight_layout()
  plt.scatter(s_train.var78, s_train.power, s=0.1)

  attributes = s_train.getAttributes()
  names = s_train.getNames()
  coef = s_train.correlation()
  for v in range(BEGIN_AT, NUMBER_ATTRIBUTES):
    for h in range(BEGIN_AT, NUMBER_ATTRIBUTES):
      if (v != h):
        axs[v][h].scatter(attributes[h], attributes[v], s=0.1, c=s_train.power, cmap=plt.cm.Greys)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)
        if v >= 2 and h>=2:
          xmin, xmax, ymin, ymax = axs[v][h].axis()
          xbar = (abs(xmax)-abs(xmin))/2.
          ybar = (abs(ymax)-abs(ymin))/2.
          axs[v][h].text(xbar, ybar, "{:.2f}".format(coef[v-2,h-2]), c='red', horizontalalignment='center', verticalalignment='center', clip_on=True)
      else:
        axs[v][h].text(0.5, 0.5, names[v], horizontalalignment='center', verticalalignment='center', clip_on=True)
        axs[v][h].xaxis.set_visible(False)
        axs[v][h].yaxis.set_visible(False)

def makeModel(_t=1e-3, _c=1.0, _e=0.1):
  model = svm.SVR(tol=_t, C=_c, epsilon=_e)
  #model.fit(s_train.data, s_train.power) #Used for current time predictions
  model.fit(s_train.data, s_train_24ahead.power) #Used for 24 hour ahead predictions
  return model

def runModel():
  for z in range(0, _NUMBERZONES):

    # Since there are multiple zones, we may need to shave off 24 hours.
    if len(s_test_24behind.zonedata[z]) > len(s_test.zonepower[z]):
      s_test_24behind.zonedata[z] = s_test_24behind.zonedata[z][:-24]
    elif len(s_test_24behind.zonedata[z]) < len(s_test.zonepower[z]):
      s_test.zonepower[z] = s_test.zonepower[z][24:]

    #y_pred_test = regr.predict(s_test.zonedata[z]) #Used for current time predictions
    y_pred_test = regr.predict(s_test_24behind.zonedata[z]) #Used for 24 hour ahead predictions
    # Scoring // Current Time or 24 Hours Ahead
    try: # Zones of equal length
      RMSE_Scores[z] = metrics.mean_squared_error(s_test.zonepower[z], y_pred_test, squared=False)
      MAE_Scores[z] = metrics.mean_absolute_error(s_test.zonepower[z], y_pred_test)
    except ValueError:
      pass

    if z == _ZONETOTEST:
      #plotPredictVsActual(s_test.zonepower[_ZONETOTEST], y_pred_test)
      #plotCurve(s_test.zonepower[_ZONETOTEST], y_pred_test, 'summer')
      pass

def printScores():
  RMSE_out = str()
  MAE_out = str()
  RMSE_avg = 0
  MAE_avg = 0
  for i in range(3):
    RMSE_out += ("\t" + truncate(RMSE_Scores[i], 6))
    RMSE_avg += RMSE_Scores[i]
    MAE_out += ("\t" + truncate(MAE_Scores[i], 6))
    MAE_avg += MAE_Scores[i]
  RMSE_out += ("\t" + truncate(RMSE_avg/3.0, 6))
  MAE_out += ("\t" + truncate(MAE_avg/3.0, 6))

  print("\n\t\t###### Scoring Metrics ######")
  print("\tZone 1\t\tZone 2\t\tZone 3\t\tOverall")
  print(f"RMSE{RMSE_out}")
  print(f"MAE{MAE_out}")

def plotPredictVsActual(act, pred):
  global PleaseShowMe

  def random_sample(array, size):
    return array[np.random.choice(len(array), size=size, replace=False)]

  def trendline(actual=True):
    if actual == True:
      z = np.polyfit(x1_results, sampleOfDifferences, 1)
      p = np.poly1d(z)
      plt.plot(x1_results, p(x1_results), "r--")
    if actual == False:
      #avgSampleVal = np.full(len(sampleOfDifferences), np.mean(sampleOfDifferences))
      bunchOfZeroes = np.full(len(sampleOfDifferences), 0)
      plt.plot(x1_results, bunchOfZeroes, "r--")
  
  def findErrorByElement():
    def percentError(x):
      if act[x] != 0:
        return abs((pred[x]-act[x])/act[x])
      else:
        return 0

    plt.ylim(0, 0.1)
    tempArray = np.empty([0])
    for i in range(0, len(act)):
      tempArray = np.append(tempArray, percentError(i))
    return tempArray

  plotinfo = {
    'x' : "Actual Power Output",
    'y' : "Predicted Power Output",
    'pct_sample' : 1
  }
  #sampleOfDifferences = random_sample(np.subtract(act,pred), int(len(act)*plotinfo['pct_sample']))
  #sampleOfDifferences = random_sample(act, pred*plotinfo['pct_sample'])
  #x1_results = np.arange(len(sampleOfDifferences))
  #plt.plot(x1_results, abs(sampleOfDifferences), lw=0.45)
  #plt.scatter(x1_results, (sampleOfDifferences), s=1.6)
  #plt.scatter(act, pred, s=0.5)
  plt.scatter(act, pred, s=0.1)
  plt.xlabel(plotinfo['x'])
  plt.ylabel(plotinfo['y'])
  #trendline(False)
  plt.plot(np.arange(2.0), np.arange(2.0), "r")

  plt.title(f"Zone {_ZONETOTEST+1} Results\nSampling {int(plotinfo['pct_sample']*100)}% of Data")
  PleaseShowMe = True

def plotCurve(act, pred, season):
  global PleaseShowMe

  days = {
    'winter' : [6950, 6950+24],
    'spring' : [8993, 8993+24],
    'summer' : [11176-240, 11176+24-240],
    'fall' : [13367-36, 13367+24-36]
  }
  actual_range = act[days[season][0] : days[season][1]]
  predicted_range = pred[days[season][0] : days[season][1]]
  x1_hours = np.arange(24)
  plt.scatter(x1_hours, actual_range, c='b', marker='x')
  plt.scatter(x1_hours, predicted_range, c='r', marker='x')
  plt.plot(x1_hours, actual_range, 'b--', label="Actual")
  plt.plot(x1_hours, predicted_range, 'r--', label="Predicted")
  plt.title(f"Zone {_ZONETOTEST+1} Prediction Results\nSeason: {season}")
  plt.xlabel("Hours")
  plt.ylabel("Power Output")
  plt.legend()

  PleaseShowMe = True

regr = makeModel(1e-4, 1.3, 0.03)
runModel()
printScores()
if PleaseShowMe:
  plt.show()

#regr = None
def helper_tuning():
  global regr
  for _tolerance in [1e-4]:
    for _regularization in range(13, 16, 1):
      for _epsilon in range(2, 5, 1):
        print(f"\nTolerance: {_tolerance}, Regularization: {_regularization/10.0}, Epsilon: {_epsilon/100.0}")
        regr = makeModel(_tolerance, _regularization/10.0, _epsilon/100.0)
        runModel()
        printScores()