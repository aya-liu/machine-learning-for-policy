'''
Functions for exploring data

Aya Liu

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import missingno as msno


def summarize_missing(data):
    '''
    '''
    print("########")
    print("Number and percentage of missing data in each column:")
    print("")

    # get number and % of NA values for each column
    nas = data.isnull().sum().to_frame()
    nas.columns = ['num_NA']
    nas['perc_NA'] = nas['num_NA']/len(data)
    display(nas[nas['num_NA'] != 0])

    # plot NA matrix
    msno.matrix(data)


def explore_cat_vars(group, cat_vars, data):
    '''
    '''
    for x in cat_vars:
        # Distribution of a categorical variable
        data.groupby(x).size().sort_values(ascending=False).plot(
            kind = 'bar')
        print("")
        print("########")
        print("")
        print("Number of observations by {}".format(x))
        plt.show()

        # Distribution of a categorical variable grouped by label
        grouped = data.groupby([group, x]).size().reset_index()
        sns.barplot(x=group, y=0, hue=x, data=grouped)
        plt.show()


def explore_num_vars(group, num_vars, data):
    '''
    Calculate and visualize distribution of numerical variables

    '''
    
    # Distribution of all numerical variables
    print('Distribution of all numerical variables')
    display(data[num_vars].describe())

    # Distribution of each numerical variables grouped by label:
    for x in num_vars:
        print("")
        print("########")
        print("")       
        print("Variable: {}".format(x))
        print("---")
        print("Distribution of {} grouped by {}".format(x, group))
        stats = summarize_by_group(group, x, data)
        display(stats)

        # Visualize distribution
        print("")
        plot_num_dist_by_group(group, x, data)

        # Calculate upper extreme and lower extreme to isolate outliers
        ue, le = get_extremes(stats)
        # Check percentage of outliers in each group
        get_outlier_perc(group, x, ue, le, data)


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


def get_extremes(stats):
    # Calculate upper extreme and lower extreme to isolate outliers

    ue = 0
    le = float('inf')
    for i in range(len(stats)):
        iqr = stats.iloc[i, 6] - stats.iloc[i, 4] # IQR = Q3 - Q1
        ue_i = stats.iloc[i, 6] + iqr * 1.5 # Upper Extreme = Q3 + 1.5 IQR
        le_i = stats.iloc[i, 4] - iqr * 1.5 # Lower Extreme = Q1 - 1.5 IQR
        if ue_i > ue:
            ue = ue_i
        if le_i < le:
            le = le_i
    return ue, le


def plot_num_dist_by_group(group, var, data):
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


def get_outlier_perc(group, var, upper_extreme, lower_extreme, data):
    '''
    Shows the percentage of number of outliers among all observations
    (count of outliers divided by count of all observations) for each group
    splitted by the target variable.

    Inputs:
        group: (str) target variable to group by
        var: (str) variable to plot
        upper_extreme: (int, float) upper extreme above which observations are 
                   considered outliers
        lower_extreme: (int, float) upper extreme below which observations are 
                   considered outliers
        data: (dataframe) a dataframe
    
    '''
    print('Upper outliers: % obs having {} > {}'.format(var, upper_extreme))
    for g in list(data.groupby(group).groups):
        label = '\t{} = {}'.format(group, g)
        perc = len(data[data[group] == g][data[var] > upper_extreme]) / \
               len(data[data[group] == g])
        print("{}: {:.2f}".format(label, perc))

    print("")

    print('Lower outliers: % obs having {} < {}'.format(var, lower_extreme))
    for g in list(data.groupby(group).groups):
        label = '\t{} = {}:'.format(group, g)
        perc = len(data[data[group] == g][data[var] < lower_extreme]) / \
               len(data[data[group] == g])
        print("{}: {:.2f}".format(label, perc))
