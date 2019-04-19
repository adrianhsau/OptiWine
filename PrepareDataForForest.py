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

Data = pd.read_csv('FormattedData.csv', sep = ",",encoding=Encoding)
Descriptions = Data['Descriptions']
print(Descriptions)
