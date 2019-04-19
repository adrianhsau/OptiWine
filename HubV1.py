import pandas as pd 
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import operator
import time
from pandas import DataFrame


#================================================================================#
#================================================================================#
#=============================Read in the Data===================================#
#================================================================================#
#================================================================================#

Data = pd.read_csv('wineTEST.txt', sep = ",",encoding='utf-8')
badDescription = Data['description'] #It is a bad description because each discription
#many words that have no true impact on the variety of the wine
Points = Data['points']
Price = Data['price']
Province = Data['province']
TasterName = Data['taster_name']
Variety = Data['variety']



#================================================================================#
#================================================================================#
#====================First Description Data Manipulation=========================#
#=====================----The Inital Clean Up -------============================#
#================================================================================#
#================================================================================#

#Initialize new description list
Description = [""]*len(badDescription)
V = [""]*len(badDescription)

#The purpose of this for loop is to account for all of the characters that Python is unable to properly read.
#This portion of the loop:[2:len(badDescription[i])+2] is meant to cut off the initial and final quotation marks
#remaining from the inital file read-in

for i in range(0,len(badDescription)):
	BD = badDescription[i]
	BDV = str(Variety[i])
	Description[i] = str(BD.encode(encoding = 'utf-8',errors="backslashreplace"))[2:len(badDescription[i])+2]
	V[i] = str(BDV.encode(encoding = 'utf-8',errors="backslashreplace"))


print(V)
time.sleep(5)


#================================================================================#
#================================================================================#
#=======================Second Description Data Manipulation=====================#
#=======Here we determine the total number of unique words in our dataset========#
#================================================================================#
#================================================================================#

#Utilize a natural language processing (NLP) that has created a list of "filler" words that are not core components
#to our sentence descriptions ie. prepositions, conjuctions.
NLPFillerList = stopwords.words('english')

#This list has been created by us and encapsulates words not a part of the inital NLP list
listOfUselessWords = ["In","now","It","A","include","and","The","isn't","This","is","a","that","while","are","out","with","It's","Some","was","all","to","and,"]

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
	words = Description[i].split()
	for word in words:
		word = word.lower()
		if word not in ExcessiveWords:

			#Clean up punctuation at the end of the word
			if word[len(word)-1] in punction:
				word = word[0:len(word)-1]

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
	words = Description[i].split()
	
	for word in words:
		#Clean up punctuation at the end of the word
		if word[len(word)-1] in punction:
			word = word[0:len(word)-1]

		#Recreate the descriptions only with the valuable terms
		if word.lower() in UniqueWords:
			SimplifiedDescription[i] = SimplifiedDescription[i] + " " + str(word)
			


#================================================================================#
#================================================================================#
#============================ Data Visualization ================================#
#== In order to properly use the random forest, we must recreate the inital =====#
#=== descriptions of the wine, but now only with the words we care about ========#
#================================================================================#
#================================================================================#


freq = nltk.FreqDist(UniqueWords)
# freq.plot(20, cumulative=False)

# plt.figure(figsize=(15,10))
# freq(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Number of Wines")
# plt.show()

#wordcloudText = " ".join(descrip for descrip in SimplifiedDescription)

# # Create stopword list:
# stopwords = set(STOPWORDS)
# stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# # Generate a word cloud image
# wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)


# #Create and generate a word cloud image:
# wordcloud = WordCloud(background_color="white").generate(wordcloudText)


# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()


#================================================================================#
#================================================================================#
#==================== Export Data for Machine Learning ==========================#
#== In order to properly use the random forest, we must recreate the inital =====#
#=== descriptions of the wine, but now only with the words we care about ========#
#================================================================================#
#================================================================================#



SetUpDictionary = {'Description':SimplifiedDescription}
df = DataFrame(SetUpDictionary)

df.to_csv("DescriptionData.csv",encoding = 'utf-8',index=False)
