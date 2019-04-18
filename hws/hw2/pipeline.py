import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


### explore

def summarize_by_group(group, var, data):
    '''
    Get summary statistics for a variable grouped by the target variable
    
    Inputs:
        group: (str) target variable to group by
        var: (str) variable to summarize
        data: (dataframe) a dataframe
    Returns: a pandas dataframe containing summary statistics
    
    '''
    return data.groupby(group)[var].describe()

def plot_dist(group, var, data):
    '''
    Plot distribution of a variable grouped by the target variable.
    One boxplot includes outliers and one excludes outliers.
    
    Inputs:
        group: (str) target variable to group by
        var: (str) variable to plot
        data: (dataframe) a dataframe

    '''
    # Distribution with outliers
    plt.subplot(1, 2, 1)
    sns.boxplot(x=group, y=var, showfliers=True, data=data)
    plt.title('With outliers')

    # Distribution without outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(x=group, y=var, showfliers=False, data=data)
    plt.title('Without outliers')
    plt.tight_layout()
    plt.show()

def perc_outliers(group, var, threshold, data):
    '''
    Shows the percentage of number of outliers among all observations
    (count of outliers divided by count of all observations) for each group
    splitted by the target variable.

    Inputs:
        group: (str) target variable to group by
        var: (str) variable to plot
        threshold: (int, float) a threshold above which observations are 
                   considered outliers
        data: (dataframe) a dataframe
    
    '''
    print('Percentage of people having {} > {}'.format(var, threshold))
    for g in list(data.groupby(group).groups):
        label = '\tfor {} = {}:'.format(group, g)
        perc = len(data[data[group] == g][data[var] > threshold]) / \
               len(data[data[group] == g])
        print(label, perc)


### pre-process

def fill_na_with_median(data):
    '''
    Fill all columns with NA values with median values of those columns

    Inputs: a dataframe
    Returns: a dataframe with no NA values

    '''
    cols_na = {col: data[col].median() for col in data.columns 
               if data[col].isna().any()}
    return data.fillna(cols_na)


### generate features

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

def convert_to_dummy(varname, data):
    '''
    Convert a categorical variable to dummy variables.

    Inputs:
        varname: (str) name of the continous variable
        data: (dataframe) the dataframe
   
    Returns: a dataframe with additional dummy columns

    '''
    df_dummies = pd.get_dummies(data[[varname]])
    df_new = pd.concat([data, df_dummies], axis=1)
    return df_new


### build classifier

def build_tree_clf(target, feat_cols, data, test_size=0.25, train_size=None,
                   random_state=None, shuffle=True, stratify=None):
    '''
    Builds and tests a decision tree classifier.

    Inputs:
        target: (str) the target variable
        feat_cols: (list) a list of feature variable names
        data: (dataframe) the dataframe

        The following optional parameters are for the train-test-split:

        (Optional) test_size: (float, int) the proportion of the dataset to 
                    include in the test split. Default=0.25.
        (Optional) train_size: (float, int) the proportion of the dataset to
                    include in the train split. If None, the value is 
                    automatically set to the complement of the test size. 
                    Default=None.
        (Optional) random_state: (int, RandomState instance or None)
                    random number generator. If None, the random number 
                    generator is the RandomState instance used by np.random.
                    Default=None.
        (Optional) shuffle: (bool) Whether to shuffle the data before splitting
                    If False, then stratify must be None. Default=True.
        (Optional) stratify: (array-like) split data in a stratified fashion
                    using this as the class labels. Defaul=None.
    
    Returns:
        (clf, y_test, y_pred): a tuple of the classifier, array of test values, 
                               array of predicted values


    '''
    # split dataset in features and target variable
    X = data[feat_cols]
    y = data[target]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
                                       X, y, 
                                       test_size=test_size, 
                                       train_size=train_size,
                                       random_state=random_state,
                                       shuffle=shuffle,
                                       stratify=stratify)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    return clf, y_test, y_pred


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



