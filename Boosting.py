import hyperopt
import lightgbm as lgb
from hyperopt import STATUS_OK
import pandas as pd
import numpy as np
import time
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin



Encoding = 'utf-8'

Data = pd.read_csv('FormattedData2.csv', sep = ",",encoding=Encoding).sort_values('Variety')

DataFormat = Data.sort_values("Variety")

Variety = DataFormat['Variety']


Descriptions = DataFormat['Description']


#Initalize a dictionary that will store the each unique word and its frequency
UniqueWords = dict()

#The purpose of this for-loop is to fill our dictionary with appropriate words and keep track of the number
#of times they appear
for i in range(0,len(Descriptions)):
	words = Descriptions[i].split()
	for word in words:
		word = word.lower()
			#Add to frequency counter of specific word
		if word in UniqueWords:
			UniqueWords[word] += 1
		else:
			UniqueWords[word] = 1

#print(UniqueWords)

#================================================================================#
#================================================================================#
#========================== Data Manipulation Number 4 ==========================#
#== We must now reformat the data to utilize dummy variables for our ML forest ==#
#================================================================================#
#================================================================================#
#================================================================================#

DescriptionVector = [""]*len(UniqueWords)
counter = 0
for key in UniqueWords:
	DescriptionVector[counter] = key
	counter += 1

UniqueVarieties = dict()
VarietyCount = 0
for i in range(0,len(Variety)):
	WineType = str(Variety[i])
	if WineType not in UniqueVarieties:
		UniqueVarieties[WineType] = VarietyCount
		VarietyCount += 1


WineVector = [""]*len(UniqueVarieties)
counter = 0
for key in UniqueVarieties:
	WineVector[counter] = key
	counter += 1


#Create a 1D Vector holding the wine variety number for each description
VarietyVectorBigly = [""]*len(Variety)
for i in range(0,len(Variety)):
	VarietyVectorBigly[i] = UniqueVarieties[Variety[i]]


WordVector = dict()
WordCount = 0
for key in UniqueWords:
	if key not in WordVector:
		WordVector[key] = WordCount
		WordCount += 1

#print(VarietyVectorBigly)

#print(VarietyVectorBigly)
SUPERMATrick = np.zeros((len(VarietyVectorBigly),len(UniqueWords)),dtype="int8")
for i in range(0,len(VarietyVectorBigly)):
	words = Descriptions[i].split()
	for word in words:
		word = word.lower()
		SUPERMATrick[i,WordVector[word]] = 1



# Labels are the values we want to predict
labels = np.array(VarietyVectorBigly[0:len(VarietyVectorBigly)],dtype = "int8")


# Remove the labels from the features
# Saving feature names for later use
# Convert to numpy array
FeaturesDescriptions = np.array(SUPERMATrick)



Nfolds = 10
print("ML BEGIN")
start = time.time()
train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

train_set = lgb.Dataset(train_features, train_labels)
model = lgb.LGBMClassifier()


start = time.time()
model.fit(train_features, train_labels)
train_time = time.time() - start


print(train_time)

predictions = model.predict_proba(test_features)[:, 1]
auc = roc_auc_score(test_labels, predictions)






