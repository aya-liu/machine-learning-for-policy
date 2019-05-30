import sys
import warnings
warnings.filterwarnings('ignore')
from dateutil import parser
import numpy as np
import pandas as pd
import pipeline as pp

if __name__ == '__main__':

	file = sys.argv[1:][0]
	output_dir = sys.argv[1:][1]
	grid_size = sys.argv[1:][2]
	debug = sys.argv[1:][3]

	print('Reading data...')
	# Read data
	coltypes = {'school_ncesid': str}
	parse_dates = ['date_posted', 'datefullyfunded']
	df = pp.read_csv(file, coltypes=coltypes, parse_dates=parse_dates)

	print('Preparing data...')
	# Prepare data
	## Generate outcome variable
	df['time_till_funded'] = (df.datefullyfunded - df.date_posted).apply(lambda x: x.days)
	df['not_funded_wi_60d'] = np.where(df.time_till_funded > 60, 1, 0)

	print('Constructing pipeline...')
	# Construct pipeline
	pipeline = pp.Pipeline()

	# Set pipeline parameters
	label = 'not_funded_wi_60d'
	predictor_sets = [['school_city', 'school_state',
	       'school_metro', 'school_district', 'school_county', 'school_charter',
	       'school_magnet', 'teacher_prefix', 'primary_focus_subject',
	       'primary_focus_area', 'secondary_focus_subject', 'secondary_focus_area',
	       'resource_type', 'poverty_level', 'grade_level',
	       'total_price_including_optional_support', 'students_reached',
	       'eligible_double_your_impact_match']]
	time_col = 'date_posted'
	start = parser.parse('2012-01-01')
	end = parser.parse('2013-12-31')
	test_window_months = 6
	outcome_lag_days = 60
	output_filename = 'evaluations.csv'
	ks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	print('Running pipeline...')
	pipeline.run(df, time_col, predictor_sets, label, start, end, test_window_months, 
            outcome_lag_days, output_dir, output_filename, grid_size=grid_size, thresholds=[], 
            ks=ks, save_output=True, debug=debug)
