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
import timeit
import re

start = time.time()
sys.stdout = codecs.getwriter( "utf-8" )( sys.stdout.detach() )



#================================================================================#
#================================================================================#
#=============================Read in the Data===================================#
#================================================================================#
#================================================================================#

Encoding = 'utf-8'
Data = pd.read_csv('FullWineList.txt', sep = ",",encoding=Encoding)
badDescription = Data['description'] #It is a bad description because each discription
#many words that have no true impact on the variety of the wine
Points = Data['points']
Price = Data['price']
Province = Data['province']
TasterName = Data['taster_name']
Variety = Data['variety']



#================================================================================#
#================================================================================#
#==================== First Description Data Manipulation =======================#
#==== Initial clean up of words that we have deemed useless, along with =========#
#============= stopwords such as prepositions and conjunctions ==================#
#================================================================================#

#Initialize new description list
Description = [""]*len(badDescription)
V = [""]*len(badDescription)
T = [""]*len(badDescription)
P = [""]*len(badDescription)

#The purpose of this for loop is to account for all of the characters that Python is unable to properly read.
#This portion of the loop:[2:len(badDescription[i])+2] is meant to cut off the initial and final quotation marks
#remaining from the inital file read-in

for i in range(0,len(badDescription)):
	BD = badDescription[i]
	BDV = str(Variety[i])
	BDT = str(TasterName[i])
	BDP = str(Province[i])
	Description[i] = str(BD.encode(encoding = Encoding,errors="backslashreplace"))[2:len(badDescription[i])+2]
	V[i] = str(BDV.encode(encoding = Encoding,errors="backslashreplace"))[2:len(badDescription[i])+2]
	T[i] = str(BDT.encode(encoding = Encoding,errors="backslashreplace"))[2:len(badDescription[i])+2]
	P[i] = str(BDP.encode(encoding = Encoding,errors="backslashreplace"))[2:len(badDescription[i])+2]


#================================================================================# 
#=======================Second Description Data Manipulation=====================#
#=======Here we determine the total number of unique words in our dataset========#
#================================================================================#
#================================================================================#

#Utilize a natural language processing (NLP) that has created a list of "filler" words that are not core components
#to our sentence descriptions ie. prepositions, conjuctions.
NLPFillerList = stopwords.words('english')

#This list has been created by us and encapsulates words not a part of the inital NLP list
listOfUselessWords = ["seems","soon","drink","aromas","many","somewhat","many","shows","along","becoming","price","come","wine","tastes","flavors","that's","In","now","It","A","include","and","The","isn't","This","is","a","that","while","are","out","with","It's","Some","was","all","to","and,"]

ExcessiveWords = NLPFillerList + listOfUselessWords

#Initalize a dictionary that will store the each unique word and its frequency
UniqueWords = dict()

#These two variables were created to help clean up any remaining string containing punctuation or the error codes
#from the NonASCII capable characters
punction = ".,:;!)"
NonASCII = ["xe2","xc3","xa9","x80","x94","xad","xa4"]


#The purpose of this for-loop is to fill our dictionary with appropriate words and keep track of the number
#of times they appear
for i in range(0,len(Description)):
	D = re.sub('[^a-zA-Z]',' ',Description[i])
	words = D.split()
	for word in words:
		word = word.lower()

		#Clean up punctuation at the end of the word
		if word[len(word)-1] in punction:
				word = word[0:len(word)-1]

		if word not in ExcessiveWords:
			#Eliminate Words with NonASCII characacters
			for badWord in NonASCII:
				if badWord in word:
					word = "null"

			#Add to frequency counter of specific word
			if word in UniqueWords:
				UniqueWords[word] += 1
			else:
				UniqueWords[word] = 1



#Remove words with nonASCII Characters
del UniqueWords["null"]


#Determine the number of words that are only used as descriptors once
MoreUselesswords = [""]*len(UniqueWords)
index = 0
for key,value in UniqueWords.items():
	if UniqueWords[key] == 1:
		MoreUselesswords[index] = key
		index += 1


#Remove empty strings from array
MoreUselesswords = list(filter(None, MoreUselesswords))



#================================================================================#
#================================================================================#
#=======================Third Description Data Manipulation======================#
#== In order to properly use the random forest, we must recreate the inital =====#
#=== descriptions of the wine, but now only with the words we care about ========#
#================================================================================#
#================================================================================#

#Initialize the new wine description list
SimplifiedDescription = [""]*len(badDescription)

#Determine number of sentences per description and create list only with words that are actually usefull
for i in range(0,len(Description)):
	D = re.sub('[^a-zA-Z]',' ',Description[i])
	words = D.split()
	for word in words:
		#Recreate the descriptions only with the valuable terms
		if word.lower() in UniqueWords and word.lower() not in MoreUselesswords:
			SimplifiedDescription[i] = SimplifiedDescription[i] + " " + str(word)


#Modify the Variety, Province, and Taster Entries to eliminate any ending punction
for i in range(0,len(V)):
	V[i] = V[i][0:len(V[i])-1]
	T[i] = T[i][0:len(T[i])-1]
	P[i] = P[i][0:len(P[i])-1]

#================================================================================#
#================================================================================#
#==================== Export Data for Further Processing ========================#
#================================================================================#
#================================================================================#
#================================================================================#
#================================================================================#

SetUpDictionary = {'Description':SimplifiedDescription,'Variety':V,'Province':P,'TasterName':T}
df = DataFrame(SetUpDictionary)

df.to_csv("FormattedData.csv",encoding = Encoding,index=False)