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
import csv
from hyperopt.pyll.stochastic import sample
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
import warnings
import ast
warnings.filterwarnings('ignore')

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

#Split the data into testing and training sets
train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

print("Read in CSV")
results = pd.read_csv('gbm_trialsFinals.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)

# Convert from a string to a dictionary
ast.literal_eval(results.loc[0, 'params'])

# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()


del best_bayes_params['metric']

print("Creating Model")
# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMModel(n_estimators=best_bayes_estimators, n_jobs = -1, metric = 'multi_error',
                                    random_state = 50, **best_bayes_params)

print("Fitting Model")
best_bayes_model.fit(train_features, train_labels)


print("Predicting Model")
# Evaluate on the testing data 
Predictions = best_bayes_model.predict(test_features)


correct = 0
#Calculate the number of times the model correctly predicted the test labels given the test features
for i in range(0,Predictions.shape[0]):
    maxProbability = np.max(Predictions[i,:])
    for j in range(0,len(Predictions[i,:])):
        if Predictions[i,j] == maxProbability:
            WinePredictionss = j
            break
    if WinePredictionss == test_labels[i]:
        correct += 1

TotalError = 1 - correct/len(test_labels)

print('The best model from Bayes optimization scores {:.5f} error on the test set.'.format(TotalError))