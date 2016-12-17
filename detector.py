import pandas
import csv
import numpy as np
import sys
import string
import unicodedata

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))
stop_words = stopwords.words("english")

def tokenizer(message):
	message = unicode(message,'utf-8').lower()
	message = remove_punctuation(message)
	words = [ word for word in word_tokenize(message) if word not in stop_words ]
	WNL = WordNetLemmatizer()
	return [ WNL.lemmatize(word) for word in words ]

def remove_punctuation(string):
	return string.translate(table)

def load_dataset():
	try:
		data = pandas.read_csv("spam_dataset", sep="\t", names=["label", "message"])
		data["length"] = data["message"].map(lambda sms: len(sms))
		return data

	except Exception as e:
		print e
		sys.exit(1)