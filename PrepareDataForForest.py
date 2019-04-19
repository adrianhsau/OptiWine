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

Encoding = 'utf-8'

Data = pd.read_csv('FormattedData.csv', sep = ",",encoding=Encoding).sort_values('Variety')
Descriptions = Data['Description']
Variety = Data['Variety']

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



VarietyVectorBigly = [""]*len(Variety)
for i in range(0,len(Variety)):
	VarietyVectorBigly[i] = UniqueVarieties[Variety[i]]



WordVector = dict()
WordCount = 0
for key in UniqueWords:
	if key not in WordVector:
		WordVector[key] = WordCount
		WordCount += 1



SUPERMATrick = np.zeros((len(VarietyVectorBigly)+1,len(UniqueWords)+1))
for i in range(0,len(VarietyVectorBigly)):
	words = Descriptions[i].split()
	for word in words:
		word = word.lower()
		SUPERMATrick[i,WordVector[word]] = 1

df = DataFrame(SUPERMATrick)



#df.to_csv("ML.csv",encoding = Encoding,index=False)

