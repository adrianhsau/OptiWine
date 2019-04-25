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
import ast
import shap
import matplotlib.pyplot as plt
import warnings
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
start = time.time()
train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

lgb_train = lgb.Dataset(train_features, label=train_labels,categorical_feature= "auto")
lgb_test = lgb.Dataset(test_features, test_labels,reference =lgb_train,categorical_feature= "auto")

print("Read in hyperparameters from hypertuning ")
# results = pd.read_csv('gbm_trials2.csv')

# # Sort with best scores on top and reset index for slicing
# results.sort_values('loss', ascending = True, inplace = True)
# results.reset_index(inplace = True, drop = True)

# # Convert from a string to a dictionary
# ast.literal_eval(results.loc[0, 'params'])

# # Extract the ideal number of estimators and hyperparameters
# best_bayes_estimators = int(results.loc[0, 'estimators'])
# best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()


lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': len(UniqueVarieties),
    'metric': ['multi_error'],
    "learning_rate": 0.05,
     "num_leaves": 60,
     "max_depth": 9,
     "feature_fraction": 0.45,
     "bagging_fraction": 0.3,
     "reg_alpha": 0.15,
     "reg_lambda": 0.15,
#      "min_split_gain": 0,
      "min_child_weight": 0,
                }


best_bayes_params = lgbm_params
#================================================================================#
#================================================================================#
#=============================== Model Training =================================#
#== We will train our model using the TPE (Tree-Structured Parzen Estimator) ====#
#====== algorithm. The parameters used in the model were found from the  ========#
#======================== HyperparameterTuning.py file. =========================#
#================================================================================#

print("Training")
print(" ")

booster = lgb.train(best_bayes_params,lgb_train,num_boost_round=500,valid_sets=[lgb_train,lgb_test],early_stopping_rounds=10,feature_name=[str(key) for key in UniqueWords])
end = time.time()
TotalTime = end - start
print('The baseline training time is {:.4f} seconds'.format(TotalTime))
print(" ")



#================================================================================#
#================================================================================#
#=============================== Model Prediction ===============================#
#== We can now use the trained model to predict the wine varieties (test_labels) #
#=================== based on our descriptors (test_features) ===================#
#================================================================================#
#================================================================================#

print("Predicting")
print(" ")
Predictions=booster.predict(test_features, num_iteration=booster.best_iteration)
print(" ")

correct = 0
#Calculate the number of times the model correctly predicted the test labels given the test features
for i in range(0,Predictions.shape[0]):
    maxProbability = np.max(Predictions[i,:])
    for j in range(0,len(Predictions[i,:])):
        if Predictions[i,j] == maxProbability:
            WinePredictions = j
            break
    if WinePredictions == test_labels[i]:
        correct += 1

TotalError = 1 - correct/len(test_labels)

print('The best model from Bayes optimization scores {:.5f} error on the test set.'.format(TotalError))


print('Plotting feature importances...')
ax = lgb.plot_importance(booster, max_num_features=20)
plt.show()