'''
Utility unctions to analyze and plot crime data.

'''

import pandas as pd
import geopandas as gpd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

### Functions for summary stats of crimes data

def groupby_major_crime_types(data, threshold, fil=None):
    '''
    Find the number and percentage of crimes for each major crime type.
    
    Inputs:
        - data: (DataFrame) crime dataset.
        - threshold: (float) a percentage threshold (e.g. 0.03) that determines
                             which crime types to bundle into "other"
        - (Optional) fil: initial filter for the dataset (e,g. year == '2017')

    Returns: a dataframe containing type breakdown of the crime data
    '''
    if fil is not None:
        data = data[fil]
        
    by_type = data.primary_type.value_counts().to_frame().reset_index().\
              rename(columns={"index":"Crime Type", "primary_type":"Number"})
    by_type["Percent"] = by_type.Number / len(data)
    top_types = by_type[by_type.Percent > threshold]
    
    # Combine crime types that make up a fraction of total crimes that is 
    # below the threshold into "Other"

    new_row = len(top_types)
    top_types.loc[new_row, 'Crime Type']  = 'OTHER'
    top_types.loc[new_row, 'Number']  = sum(by_type[by_type.Percent < \
                                                    threshold].Number)
    top_types.loc[new_row, 'Percent'] = top_types.loc[new_row, 'Number'] / \
                                        sum(top_types.Number)
    
    top_types = top_types.astype({'Number': int})

    return top_types


def plot_donut_chart(data, labels, sizes, title=None, colors=None):
    '''
    Plot donut chart for data.
    
    Inputs:
        - data: a Pandas Dataframe
        - labels: (str) column name of the dataframe as donut chart labels
        - sizesL (str) column name of the dataframe as donut chart sizes
        - (Optional) title: (str) title of the chart
        - (Optional) colors: (list) list of hex values as color palette

    '''
    labels = data[labels]
    sizes = data[sizes]
    explode = [0.05] * len(data)

    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,  
            startangle=90, pctdistance=0.85, explode = explode)

    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    if title:
        fig = plt.title(label=title, fontdict = {'fontsize': 16,
              'fontweight' : 1}, loc = 'left', pad = 30)

    ax1.axis('equal')
    plt.show()


### Functions for analyzing block characteristics for a crime type


def get_crime_type_data(crime_type, crimes_data, year=None):
    '''
    Get crime data of a specific crime type in a specific year (optional).

    Input: 
        - crime_type: (str) a crime type in upper cases
        - crimes_data: (DataFrame) the crimes dataframe
        - (optional) year: (str) year
    Returns: a Pandas DataFrame

    '''
    crime_type_data = crimes_data[crimes_data.primary_type == crime_type]
    if year:
        crime_type_data = crime_type_data[crime_type_data.year == year]
    return crime_type_data


def get_top_blocks(crime_type_data):
    '''
    Get block groups and the corresponding numbers of crime reports in the 
    blocks, sorted from the highest number to the lowest.

    Input:
        - crime_type_data: (DataFrame) a crimes dataset

    Returns: a Pandas DataFrame of block group GEOIDs and crime counts 

    '''
    top_blocks = crime_type_data.groupby('GEOID').size(). \
                 sort_values(ascending=False).reset_index()
    top_blocks.columns = ['GEOID', 'count']

    # create a table of population by GEOID
    # https://stackoverflow.com/a/35268906/1281879
    pops = crime_type_data.groupby(['GEOID','population']).size().reset_index().rename(columns={0:'number'})

    top_blocks = top_blocks.merge(pops, on='GEOID', how='inner')

    return top_blocks


def map_blocks(top_blocks, blocks_geodata, normalizeByPopulation=True):
    '''
    Generate heat map of crime report counts at block-group level.

    Input:
        - top_blocks: DataFrame of block group GEOIDs and crime counts 
        - blocks_geodata: DataFrame containing block roup GEOIDs and 
                          their geometry information
        - normalizeByPopulation: (bool) whether or not to normalize the plot by population

    '''
    # Merge top blocks with geographical information
    top_blocks = top_blocks.merge(blocks_geodata, on='GEOID', how='left')

    # normalize by population
    if normalizeByPopulation:
        # remove rows where count or population is 0.
        # https://stackoverflow.com/a/27020741/1281879
        top_blocks = top_blocks[(top_blocks != 0).all(1)]

        top_blocks['measure'] = top_blocks['count']/top_blocks['population']
        plot_title = 'Crimes per person')
    else:
        top_blocks['measure'] = top_blocks['count']
        plot_title = 'Number of crimes')

    # Convert to geodataframe
    top_blocks = gpd.GeoDataFrame(top_blocks, geometry='geometry')

    # Make heat map of number of crimes
    top_blocks.plot(figsize=(8, 6), column='measure', cmap='OrRd', legend=True, norm=matplotlib.colors.LogNorm())

    plt.title(plot_title)

    plt.show()


def describe_blocks(crime_type, top_blocks, census, outliers=None, plot=True):
    '''
    Calculate summary stats on block group ACS variables and (optionally) plot
    their correlation with the number of crimes of a specific type in the 
    block.

    Inputs:
        - crime_type: (str) type of crime
        - top_blocks: DataFrame of block group GEOIDs and crime counts 
        - census: DataFrame containing ACS variables
        - (optional) outliers: (list) list of GEOIDs to drop out of the data
        - (optional) plot: (bool) whether shows scatter plots of the number 
                    of crimes with the ACS variables. Default is True.

    '''

    # Drop outlier
    if outliers:
        top_blocks = top_blocks[~top_blocks.GEOID.isin(outliers)]

    # Get ACS variable for blocks with most battery reports
    top_blocks = top_blocks.merge(census, on='GEOID', how='left')
    # Drop rows with no median income value
    top_blocks = top_blocks[top_blocks.median_income > 0]

    # Generate mean and sd of ACS variables
    print(top_blocks.dropna().describe()[['population', 'median_income', 
                                           'owner_occupancy_rate']][1:])
    if plot:
        # Plot relationships betwen number of reports and ACS variables
        sns.lmplot(x="population", y="count", data=top_blocks)
        plt.title('{} and block population'.format(crime_type))
        sns.lmplot(x="median_income", y="count", data=top_blocks)
        plt.title('{} and block median income'.format(crime_type))
        sns.lmplot(x="owner_occupancy_rate", y="count", data=top_blocks)
        plt.title('{} and block owner occupancy rate'.format(crime_type))

        plt.show()






