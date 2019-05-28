'''
pre-processing functions

(dataset-specific)
'''

import pandas as pd

def preprocess(X, y):
	'''
	preprocess data
	'''
	# drop rows with NA values for certain columns
	df = pd.concat([X, y], axis=1)
	cols_to_drop_na_rows = ['school_district', 
	                'primary_focus_subject', 
	                'primary_focus_area', 
	                'resource_type', 
	                'grade_level']
	df.dropna(axis=0, how='any', subset=cols_to_drop_na_rows, inplace=True)
	y = df.iloc[:,-1]
	X = df.iloc[:,0:-1]

	# fill specific columns with median
	X['students_reached'].fillna(value=X['students_reached'].median())

	# encode

	# scale

def predictors_to_discretize():
	rv = {'students_reached': (
								[0, 20, 30, 90, float('inf')],
								['<20', '20-30', '30-90', '>90']),
		   'total_price_including_optional_support':(
								[0, 300, 600, 900, 1200, float('inf')],
								['<300', '300-600', '600-900', '900-1200', '>1200'])}
	return rv