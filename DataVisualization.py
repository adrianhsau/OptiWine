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

DataDes = pd.read_csv('FormattedData.csv', sep = ",",encoding='utf-8')

Descriptions = DataDes['Description']



#================================================================================#
#================================================================================#
#============================ Data Visualization ================================#
#== In order to properly use the random forest, we must recreate the inital =====#
#=== descriptions of the wine, but now only with the words we care about ========#
#================================================================================#
#================================================================================#


#freq = nltk.FreqDist(DescriptionCloud)
#freq.plot(20, cumulative=False)

# plt.figure(figsize=(15,10))
# freq(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Number of Wines")
# plt.show()

wordcloudText = " ".join(descrip for descrip in Descriptions)

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(wordcloudText)


# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
