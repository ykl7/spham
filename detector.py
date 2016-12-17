import pandas
import csv
import numpy as np
import sys

def load_dataset():
	try:
		data = pandas.read_csv("spam_dataset", sep="\t", names=["label","message"])
		data["length"] = data["message"].map(lambda sms: len(sms))
		return data

	except Exception as e:
		print e
		print 'Error: Dataset not loaded.'
		sys.exit(1)