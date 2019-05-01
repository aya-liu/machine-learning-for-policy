
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import explore
import learn
import evaluate


##### read #####

def read_data(data_file, coltypes=None, parse_dates=None):
	'''
	Read csv file into a dataframe.

	'''
	df = pd.read_csv(data_file, dtype=coltypes, parse_dates=parse_dates)
	return df


##### explore #####

def explore_data(group, num_vars, cat_vars, data):
    '''
    Explore distribution of numerical and categorical variables.
    Show tables and plots.

    '''
    explore.explore_num_vars(group, num_vars, data)
    explore.explore_cat_vars(group, cat_vars, data)
    explore.summarize_missing(data)


##### pre-process #####

def summarize_na(data):

    # get number and % of NA values for each column
    nas = data.isnull().sum().to_frame()
    nas['perc'] = nas[0]/len(data)
    display(nas[nas[0] != 0])

    # plot NA matrix
    msno.matrix(data)


def fill_na_with_median(data):
    '''
    Fill all columns with NA values with median values of those columns

    Inputs: a dataframe
    Returns: a dataframe with no NA values

    '''
    cols_na = {col: data[col].median() for col in data.columns 
               if data[col].isna().any()}
    return data.fillna(cols_na)


##### generate features #####

def visualize_num_distr(varname, target, data, bins):
    '''
    for determining bins
    '''
    data.hist(column=varname, bins=bins)
    plt.title("All observations")
    for g in data.groupby(target).groups:
        data[data[target] == g].hist(column=varname, bins=bins)
        plt.title("{} = {}".format(target, g))
    plt.show()


def discretize(varname, data, bins, labels=None):
    '''
    Convert a continous variable to a categorical variable.

    Inputs:
        varname: (str) name of the continous variable
        data: (dataframe) the dataframe
        bins: (int, sequence of scalars, or pandas.IntervalIndex)
              the criteria to bin by. 
        (Optional) labels: (array or bool) specifies the labels for the 
                           returned bins. Must be the same length as the 
                           resulting bins. If False, returns only integer 
                           indicators of the bins.
    Returns: a pandas Series of the categorical variable

    '''
    return pd.cut(data[varname], bins=bins, labels=labels)


def convert_to_dummy(data, cols_to_convert, dummy_na=True):
    '''
    Convert a list of categorical variables to dummy variables.

    Inputs:
        cols_to_convert: (list) list of variable names
        data: (dataframe) the dataframe
        dummy_na: (bool)
   
    Returns: a dataframe containing dummy variables of cols_to_convert

    '''
    return pd.get_dummies(data, dummy_na=dummy_na, columns=cols_to_convert)


### build classifier





### evaluate classifer

def evaluate_accuracy(y_test, y_pred):
    '''
    Calculates the model accuracy score.
    
    Input:
        y_test: (array)  test set of data points
        y_pred: (array) predicted values of data points

    Returns (float): model accuracy score
    
    '''

    return metrics.accuracy_score(y_test, y_pred)



