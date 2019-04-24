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
#warnings.filterwarnings('Warnings')

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
print(" ")

train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print(" ")


#Determine the number of times the model will cross-validate
n_folds = 2

# Create the dataset
train_set = lgb.Dataset(train_features, train_labels)


def objective(params, n_folds = n_folds):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    out_file = 'gbm_trials.csv'

    # Keep track of evals
    global iteration
    
    iteration += 1
    
    
    start = time.time()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 250, nfold = n_folds, 
                        early_stopping_rounds = 10, metrics = 'multi_error', seed = 50)
    
    run_time = time.time() - start
    
    # Extract the best score
    bestScore = np.min(cv_results['multi_error-mean'])
    
    # Loss must be minimized
    loss = bestScore
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmin(cv_results['multi_error-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, iteration, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': iteration,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}


#This is where we determine the hyperparameters that will be used in our model. Our goal is to allow
#TPE to choose the parameters and then do a number of evaluations for each set of spaces. We will then choose
#the space that resulted in the lowest amount of error. Then we will you that space as the parameters for our
#main boosting phase.

space =  {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(UniqueVarieties),
    'metric': ['multi_error'],
    "max_depth": hp.choice('max_depth',np.arange(15,30,dtype=float)),
    "feature_fraction": hp.choice('feature_fraction',np.arange(0.1,0.7,dtype=float)),
    'num_leaves': hp.choice('num_leaves',np.arange(1,75,dtype=int)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    
                 }

#Create a CSV file to save the results from each round
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

#Write the columns of the csv
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

#The TPE.Suggest algorithm utilizes
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

global  iteration
iteration = 0

#The number of max evals determines the overall number of iterations the minimizer will perform. The number of
#evals is also equal to the number of different spaces. Our goal is to find the best space.
maxEvals = 10

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = maxEvals, trials = bayes_trials,rstate = np.random.RandomState(50))


