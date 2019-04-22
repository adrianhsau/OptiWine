import hyperopt
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
from hyperopt.pyll.stochastic import sample
import csv






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

train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)


N_FOLDS = 10

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = time.time()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    
    run_time = time.time() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}



# boosting type domain 
boosting_type = {'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

# Draw a sample
params = sample(boosting_type)

# Retrieve the subsample if present otherwise set to 1.0
subsample = params['boosting_type'].get('subsample', 1.0)

# Extract the boosting type
params['boosting_type'] = params['boosting_type']['boosting_type']
params['subsample'] = subsample

# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}



# Keep track of results
bayes_trials = Trials()

# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

from hyperopt import tpe

MAX_EVALS = 10


# optimization algorithm
tpe_algorithm = tpe.suggest


from hyperopt import fmin

# Global variable
global  ITERATION

ITERATION = 0

print("ML YEET")
# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))


results = pd.read_csv('gbm_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()