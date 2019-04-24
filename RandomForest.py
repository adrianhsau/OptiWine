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


#Allocate each unique word in our dictionary into a vector of descriptions
DescriptionVector = [""]*len(UniqueWords)
counter = 0
for key in UniqueWords:
    DescriptionVector[counter] = key
    counter += 1


#Determine the number of unique varieties
UniqueVarieties = dict()
VarietyCount = 0
for i in range(0,len(Variety)):
    WineType = str(Variety[i])
    if WineType not in UniqueVarieties:
        UniqueVarieties[WineType] = VarietyCount
        VarietyCount += 1


#Transform the dictionary into a vector of strings
WineVector = [""]*len(UniqueVarieties)
counter = 0
for key in UniqueVarieties:
    WineVector[counter] = key
    counter += 1


#Create a 1D Vector holding the wine variety number for each description
FullWineVectorForEachDescription = [""]*len(Variety)
for i in range(0,len(Variety)):
    FullWineVectorForEachDescription[i] = UniqueVarieties[Variety[i]]


#Create a vector holding the unique wine descriptors
WordVector = dict()
WordCount = 0
for key in UniqueWords:
    if key not in WordVector:
        WordVector[key] = WordCount
        WordCount += 1


#Features matrix is where we go through each wine description and if a word appears in that description,
#it will recieve a value of 1 for that particular wine. If it does not appear, then that value will remain 0.
#Ie. if the first variety of wine was described as fruity,nutty,tart then the column value corresponding to each
#of those descriptors in the first row will be set to 0. This will help tell our model what words are being used
#to describe each wine
FeaturesMatrix = np.zeros((len(FullWineVectorForEachDescription),len(UniqueWords)),dtype="int8")
for i in range(0,len(FullWineVectorForEachDescription)):
    words = Descriptions[i].split()
    for word in words:
        word = word.lower()
        FeaturesMatrix[i,WordVector[word]] = 1


#Assign our values to labels and features, as per proper ML jargon
labels = np.array(FullWineVectorForEachDescription[0:len(FullWineVectorForEachDescription)],dtype = "int16")
FeaturesDescriptions = np.array(FeaturesMatrix)




#================================================================================#
#================================================================================#
#=============================== Machine Learning ===============================#
#================================================================================#
#================================================================================#

print("ML BEGIN")
start = time.time()
train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)

print(" ")
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print(" ")


print("Create Random Forest Model")
#Create random forest model
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=250, n_jobs=2, oob_score=False, random_state=42,
            verbose=0, warm_start=False)

print("Fitting")
rf.fit(train_features, train_labels);
print(" ")

print("Predicting")

#Use the random forest model to predict the test labels using the test features
predictions = rf.predict(test_features)
print(" ")


#Determine the number of correct answers from the prediction model
NumberCorrect = 0
for i in range(0,len(predictions)):
	if predictions[i] == test_labels[i]:
		NumberCorrect += 1

ErrorRate = 1 - NumberCorrect / len(predictions)

end = time.time()
TotalTime = end - start

print('The baseline error rate is {:.4f} seconds'.format(ErrorRate))
print('The baseline training time is {:.4f} seconds'.format(TotalTime))

