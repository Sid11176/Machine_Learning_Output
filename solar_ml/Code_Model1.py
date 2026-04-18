#CODE FOR MODEL 1
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np


#Selecting Features
features = ['DC_POWER', 'EFFICIENCY_RATIO', 'TEMP_DERATING', 'IRRADIATION']

#Dropping rows with NaN (especially TEMP_DERATING from P-RATED)
