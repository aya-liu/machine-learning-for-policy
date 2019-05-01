'''
Build classifiers for the ML pipeline

Aya Liu
04/28/2019
'''

import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import model_selection
from sklearn import metrics

### split data ###

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


def fit_and_predict(clf, x_train, y_train, x_test):
    '''
    '''
    clf = clf.fit(x_train, y_train) # train
    pred_scores = clf.predict_proba(x_test) # test
    return pred_scores


def construct_classfiers(clf_types):
    '''
    Input: clf_types: a list of classfier constructors
    Returns (dict): {classifier name: classifer object}
    '''
    clfs = {}
    for t in clf_types:
        clfs[t.__name__] = t()
    return clfs  

def iterate_over_params(params, clf):
    '''
    params: list of parameter (name, value) pairs

    '''
    pass


def iterate_over_clfs(classifiers):
    '''
    '''
    pass

### evaluate models ###

def get_eval_metric(pred_scores, y_test, metrics, population_percent=None, 
                    thresholds=None):

    print("The true number of YES is {}/{} from the data, \
          with percentage {:.2f}%\n".format(sum(y_test), len(y_test), \
          100.*sum(y_test)/len(y_test)))
    
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

### wrapper functions for sklearn classifer constructors


def construct_kNN(n_neighbors=5, weights='uniform', algorithm='auto', 
    leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, 
    **kwargs):
    '''
    Wrapper function for the sklearn.KNeighborsClassifier constructor

    Inputs: See https://scikit-learn.org/stable/modules/generated
                /sklearn.tree.KNeighborsClassifier.html
    Returns: a K-Nearest Neighbors classifer

    '''
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, 
        algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, 
        metric_params=metric_params, n_jobs=n_jobs)


def construct_DecisionTree(criterion='gini', splitter='best', max_depth=None, 
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    max_features=None, random_state=None, max_leaf_nodes=None, 
    min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, 
    presort=False):
    '''
    Wrapper function for the sklearn.DecisionTreeClassifer constructor

    Inputs: See https://scikit-learn.org/stable/modules/generated
                /sklearn.tree.DecisionTreeClassifier.html
    Returns: a DecisionTree classifer

    '''
    return DecisionTreeClassifier(criterion=criterion, splitter=splitter, 
          max_depth=max_depth, min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf, 
          min_weight_fraction_leaf=min_weight_fraction_leaf,
          max_features=max_features, random_state=random_state,
          max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
          min_impurity_split=min_impurity_split, class_weight=class_weight,
          presort=presort)


def construct_LogisticReg():
    return clf

def construct_SVM():
    return clf

def construct_RandomForest():
    return clf

def boost():
    return clf

def bag():
    return clf





