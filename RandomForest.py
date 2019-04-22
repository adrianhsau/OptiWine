import pandas as pd 
import numpy as np
from os import path
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import operator
import time
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



#================================================================================#
#================================================================================#
#=============================Read in the Data===================================#
#================================================================================#
#================================================================================#

BigBoiMatrix = pd.read_csv('ML.csv', sep = ",",encoding='utf-8')
WineVector = pd.read_csv('FullWineVectors.csv', sep = ",",encoding='utf-8')

print(BigBoiMatrix)
time.sleep(5)
print(DataFormatted)
# Labels are the values we want to predict
labels = np.array(Variety)



# Remove the labels from the features
# Saving feature names for later use
# Convert to numpy array
FeaturesDescriptions = np.array(Descriptions)

train_features, test_features, train_labels, test_labels = train_test_split(FeaturesDescriptions, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

