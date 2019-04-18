import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


### explore

def summarize_by_group(group, var, data):
    return data.groupby(group)[var].describe()

def plot_dist(group, var, data):
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
    print('Percentage of people having {} > {}'.format(var, threshold))
    for g in list(data.groupby(group).groups):
        label = '\tfor {} = {}:'.format(group, g)
        perc = len(data[data[group] == g][data[var] > threshold]) / \
               len(data[data[group] == g])
        print(label, perc)


### pre-process

def fill_na_with_median(data):
    cols_na = {col: data[col].median() for col in data.columns 
               if data[col].isna().any()}
    return data.fillna(cols_na)


### generate features

def discretize(varname, data, bins=None, labels=None):
    if bins:
        return pd.cut(data[varname], bins=bins, labels=labels)
    else:
        return pd.cut(data[varname], bins=3, labels=labels)

def convert_to_dummy(varname, data):
    df_dummies = pd.get_dummies(data[[varname]])
    df_new = pd.concat([data, df_dummies], axis=1)
    return df_new


### build classifier

def build_tree_clf(target, feat_cols, data, test_size=0.25, train_size=None,
                   random_state=None, shuffle=True, stratify=None):
    # split dataset in features and target variable
    X = data[feat_cols]
    y = data[target]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
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

def evaluate_accuracy(y_test, y_pred, verbose=True):
    return metrics.accuracy_score(y_test, y_pred)
    



