import pandas as pd 
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk


Data = pd.read_csv('wine.txt', sep = ",",encoding='utf-8')
badDescription = Data['description']
Points = Data['points']
Price = Data['price']
Province = Data['province']
TasterName = Data['taster_name']
Variety = Data['variety']


Description = [""]*len(badDescription)

for i in range(0,len(badDescription)):
	BD = badDescription[i]
	Description[i] = str(BD.encode(encoding = 'utf-8',errors="backslashreplace"))[2:len(badDescription[i])+2]

#
# Create function to extract counter of each word
#


UniqueWords = dict()
for i in range(0,len(Description)):
	words = Description[i].split()
	for word in words:
		if word in UniqueWords:
			UniqueWords[word] += 1
		else:
			UniqueWords[word] = 1

print(UniqueWords["a"])


