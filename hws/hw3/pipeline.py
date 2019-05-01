'''
Aya Liu
'''


from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import explore_utils


########## read ##########

def read_data(data_file, coltypes=None, parse_dates=None):
	'''
	Read csv file into a dataframe.

	'''
	df = pd.read_csv(data_file, dtype=coltypes, parse_dates=parse_dates)
	return df


########## explore ##########

def explore_data(group, num_vars, cat_vars, data):
    '''
    Explore distribution of numerical and categorical variables.
    Show tables and plots.

    '''
    explore.explore_num_vars(group, num_vars, data)
    explore.explore_cat_vars(group, cat_vars, data)
    explore.summarize_missing(data)


########## pre-process ##########

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


########## generate features ##########

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


########## build classifier ##########

##### split data #####

def get_feature_cols(target, data_dummies):

    # drop NaN dummies for non-NaN variables
    data_dummies = data_dummies.loc[:, (data_dummies != 0).any(axis=0)]

    # select a vector of features
    cols = data_dummies.columns.to_list()
    feat_cols = cols[cols.index(target)+1:]

    return feat_cols


def train_test_split(target, feat_cols, data, 
                     test_size=0.25, train_size=None, random_state=None,
                     shuffle=True, stratify=None):
    '''
    splits data into training and testing sets.

    Inputs:
        target: (str) the target variable
        feat_cols: (list) a list of feature variable names
        data: (dataframe) the dataframe
        optional parameters: see sklearn.model_selection.train_test_split
    
    Returns: x_train, x_test, y_train, y_test 

    '''
    # split dataset in features and target variable
    X = data[feat_cols]
    y = data[target]

    # Split dataset into training set and test set
    return model_selection.train_test_split(
                                       X, y, 
                                       test_size=test_size, 
                                       train_size=train_size,
                                       random_state=random_state,
                                       shuffle=shuffle,
                                       stratify=stratify)
def temporal_train_test_split():
    pass


##### build models #####

CLFS = {
    'DecTree': {
        'clf': DecisionTreeClassifier(),
        'paramgrid': {'max_depth': [3, 10, 20, 50]},
    },
    'KNN': {
        'clf': KNeighborsClassifier(),
        'paramgrid': {'n_neighbors': [5, 15, 50], 
                         'weights': ['uniform', 'distance']} 
    },
    'LogReg': {
        'clf': LogisticRegression(),
        'paramgrid': {'penalty': ['l1', 'l2']} 
    },      
    'RanFor': {
        'clf': RandomForestClassifier(),
        'paramgrid': {} 
    },
    'Boosting': {
        'clf': GradientBoostingClassifier(),
        'paramgrid': {} 
    },
    'Bagging': {
        'clf': BaggingClassifier(),
        'paramgrid': {} 
    },
    
}

SVM = {'clf': LinearSVC(),
       'paramgrid': {'C': [10**-1, 1 , 10]}}            


def fit_and_predict(clf, x_train, y_train, x_test, plot=False):
    '''
    '''
    clf = clf.fit(x_train, y_train) # train
    pred_scores = clf.predict_proba(x_test) # test

    if plot:
        plt.hist(pred_scores[:,1])
        plt.title("Scores on test set")
    return pred_scores


def build_classifiers(models_to_build):
    results = {}
    for model in models_to_build:


        pass




########## evaluate classifers ##########

def temporal_validation():
    pass


def get_eval_metric(pred_scores, y_test, metrics, population_percent=None, 
                    thresholds=None):

    print("Baseline: The true number of YES in test data is {}/{} ({:.2f}%)\n".format(
          sum(y_test), len(y_test), 100.*sum(y_test)/len(y_test)))
    
    if (thresholds == None) and (population_percent == None):
            raise Exception('ValueError: must have thresholds or \
                            population_percent')

    elif thresholds and population_percent:
            raise Exception('Cannot have both thresholds and \
                            population_percent')

    elif population_percent:
            # calculate score thresholds for top p% of population
            thresholds = []
            for p in population_percent:
                thresholds.append(np.percentile(pred_scores, (1-p)*100))

    # initialize dict for metric scores
    d = {'population_percent': population_percent,
         'score_threshold': thresholds}

    # calcualte evaluation metrics at various thresholds 
    for m in metrics:
        name = m.__name__[:-6]
        scores = []

        for k in thresholds:
            # assign classification at threshold k
            pred_label = [1 if x[1] > k else 0 for x in pred_scores]
            num_pred_1 = sum(pred_label)

            # calculate evaluation metric at threshold k
            score_at_k =  m(pred_label, y_test)
            scores.append(score_at_k)

        d[name] = scores
    
    results = pd.DataFrame(d)
    return results



