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



wordcloudText = " "
print("Combining reviews to create wordcloud")
for row in Descriptions:
	DesRow = row
	wordcloudText += DesRow

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors","xc","xa","xe"])

# Generate a word cloud image
print("Generating wordcloud")
wordcloud = WordCloud(stopwords = stopwords,background_color="white").generate(wordcloudText)


# Display the generated image:
print("Plotting wordcloud")
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


print("Loading Wine Bottle World Cloud")
wine_mask = np.array(Image.open("wine_mask2.png"))

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')

# Generate a wordcloud
print("Generating Cloud")
wc.generate(wordcloudText)

# store to file
wc.to_file("wineCloud.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show() 