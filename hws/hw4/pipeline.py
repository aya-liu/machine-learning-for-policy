'''
Class for a ML pipeline

Aya Liu
'''
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import data_explore as explore
import data_preprocess as preprocess


class Pipeline:
    '''
    Class for a machine learning pipeline
    '''
    def __init__(self):
        '''
        Constructor.
        '''
        self.data = None        
        self.label = None
        self.models = []
        self.grid_size = None
        self.train_test_size = []

    def load_data(self, data_filepath, coltypes=None, parse_dates=None):
        '''
        Read csv file into a dataframe.
        
        Inputs:
            data_filepath: (str) data filepath
            coltypes: (optional dict) column, data type pairs. default=None.
            parse_dates: (optional list) columns to parse as datetime 
                         objects. default=None.
        '''
        self.data = pd.read_csv(data_filepath, dtype=coltypes, parse_dates=parse_dates)

    def set_label(self, label):
        '''
        Set outcome label for pipeline.

        Input: label: (str) outcome column name
        '''
        self.label = label

    def explore_data(self, num_vars=None, cat_vars=None):
        '''
        Explore distribution of numerical and categorical variables
        and missing data. Show tables and plots.

        Inputs: 
            num_vars (optional list): column names of numerical vars
            cat_vars (optional list): column names of categorical vars
        '''
        if num_vars:
            explore.explore_num_vars(self.label, num_vars, self.data)
        if cat_vars:
            explore.explore_cat_vars(self.label, cat_vars, self.data)
        explore.summarize_missing(self.data)

    def run_temporal_loop(self, time_col, feature_cols, start, end, test_window_months, 
                          outcome_lag_days):

        pass







    #### temporal cross validation helper functions ####

    def temporal_split(self, time_col, train_start, train_end, test_start, test_end, feature_cols):
        '''
        do one train-test-split according to start/end time
        
        Inputs:
            time_col: (str) time column to split on
            train_start, train_end, test_start, test_end: (datetime) time bound for training and test set
            feature_cols: (list) feature column names

        Returns: tuple of X_train, 
        '''
        # train test split
        train_df = self.df[(self.df[time_col] >= train_start) and 
                           (self.df[time_col] <= train_end)]
        test_df = self.df[(self.df[time_col] >= test_start) and 
                          (self.df[time_col] <= test_end)]
        # split features and label
        X_train = train_df[feature_cols]
        y_train = train_df[self.label]
        X_test = test_df[feature_cols]
        y_test = test_df[self.label]
        return X_train, y_train, X_test, y_test

    def get_train_test_times(self, start, end, test_window_months, outcome_lag_days):
        '''
        start: (datetime) start time of all data
        end: (datetime) end time of all data
        test_window_months: (int/float) time length of a test set in months
        outcome_lag_days: (int) lag needed to get the outcome in days

        '''
        results = []

        # initial train/test time cutoffs
        train_start = start
        train_end = train_start + \
                    relativedelta(months=+test_window_months) - \
                    relativedelta(days=+1)
        test_start = train_end + relativedelta(days=+(outcome_lag_days+1))
        test_end =  test_start + \
                    relativedelta(months=+test_window_months) - \
                    relativedelta(days=+1)

        while test_end <= end - relativedelta(days=+outcome_lag_days):
                # save times
                results.append((train_start, train_end, test_start, test_end))

                # increment time (train_start stays the same)
                test_start = test_end + relativedelta(days=+1)
                test_end = test_start + \
                            relativedelta(months=+test_window_months) - \
                            relativedelta(days=+1)
                train_end = test_start - relativedelta(days=+(outcome_lag_days+1))
               
        # adjust test_end for the last test set:
        test_end = end - relativedelta(days=+outcome_lag_days)
        results.append((train_start, train_end, test_start, test_end))

        self.train_test_sets = results
        return results









