import pandas as pd
import numpy as np
from os import path
import nltk
from nltk.corpus import stopwords
import operator
import time
from pandas import DataFrame
import sys
import codecs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


Encoding = 'utf-8'
Data = pd.read_csv('FormattedData.csv', sep = ",",encoding=Encoding).sort_values('Variety')
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
labels = np.array(VarietyVectorBigly[0:len(VarietyVectorBigly)],dtype = "int16")
print(labels)

# Remove the labels from the features
# Saving feature names for later use
# Convert to numpy array
FeaturesDescriptions = np.array(SUPERMATrick)



print("ML BEGIN")
start = time.time()
train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=42,
            verbose=0, warm_start=False)
# Train the model on training data

print("Fitting")
rf.fit(train_features, train_labels);

print("Predicting")
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)


NumberCorrect = 0
for i in range(0,len(predictions)):
	if predictions[i] == test_labels[i]:
		NumberCorrect += 1

ErrorRate = 1 - NumberCorrect / len(predictions)

end = time.time()
TotalTime = end - start

print('The baseline error rate is {:.4f} seconds'.format(ErrorRate))
print('The baseline training time is {:.4f} seconds'.format(TotalTime))

