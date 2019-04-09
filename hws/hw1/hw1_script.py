'''
Machine Learning HW1

Aya Liu
'''
import pandas as pd
import geopandas as gpd
import geojson
from shapely.geometry import shape
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sodapy import Socrata
import analyze as an

pd.options.display.max_columns=999


# ## 1.1 Obtain data for reported crimes 2017-2018
# 
# Data Source: [Chicago Open Data Portal]
# (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/
# ijzp-q8t2)

domain = 'data.cityofchicago.org'
dataset_id = '6zsd-86xi'
client = Socrata(domain, None)
results = client.get(dataset_id, where="year in (2017, 2018)", limit=1000000)
crimes = pd.DataFrame.from_dict(results)

# convert date column to datetime and use as index
crimes.date = pd.to_datetime(crimes.date)
crimes = crimes.set_index(crimes.date)
crimes.head()


# ## 1.2 Summary statistics

# ### Types of reported crimes

top_reported = an.groupby_major_crime_types(crimes, 0.03)
top_reported.head()

# Plot of top crime types

colors = ['#415c80', '#686599', '#9b68a5', '#cf69a0', 
          '#f96e8a', '#ff826a', '#ffa344', '#ffca1c']
an.plot_donut_chart(top_reported, "Crime Type", "Number", 
                    title = "Types of Reported Crimes in Chicago, 2017-18",
                    colors=colors)

# ### Types of arrests

arrested = crimes.arrest == True
top_arrested = an.groupby_major_crime_types(crimes, 0.03, fil=arrested)
top_arrested.head()

# Total arrests
len(crimes[arrested])

# Percentage of reported crimes that resulted in arrests
len(crimes[arrested])/len(crimes)

# Percentage of top 5 arrest types
sum(top_arrested[:5].Percent)

# Plot
an.plot_donut_chart(top_arrested, "Crime Type", "Number", 
                    title = "Types of Reports that Resulted in Arrests in \
                            Chicago, 2017-18", 
                    colors=colors)


# ### Time trends

# Total number of crimes reported in 2017 and 2018
crimes.groupby("year").size()

# Number of top crime reports in 2017 and 2018
crimes.groupby(['primary_type','year']).size().sort_values(
       ascending=False).head(10)

# Prepare data for plotting reported crimes

# Create dummy variable for all reported crimes
crimes["reported"] = True
reported = crimes["reported"]

# Create dummy variables for crime types
type_dum = pd.get_dummies(crimes['primary_type'])

# Keep the dummy variables for top 5 types of reported crimes
top_rep_types = list(top_reported['Crime Type'].head())
type_dum_rep = type_dum[top_rep_types]

# Plot monthly counts of reported crimes

# All reported crimes
sns.set(style="darkgrid")
reported.resample('M').sum().plot(figsize=(9,5))
plt.title('Number of All Reported Crimes in Chicago, 2017-18', 
          fontdict = {'fontsize': 14}, loc = 'left')

# Top 5 types of reported crimes
type_dum_rep.resample('M').sum().plot(figsize=(9,5)).\
             legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Number of Top 5 Types of Reported Crimes in Chicago, 2017-18',
         fontdict = {'fontsize': 14}, loc = 'left')
plt.show()


# Prepare data for plotting arrests
# Keep the dummy variables for top 5 types of arrests
top_arr_types = list(top_arrested['Crime Type'].head())
type_dum_arr = type_dum[top_arr_types]


# Plot monthly counts of arrests
arrests = crimes[crimes['arrest'] == True]['arrest']

# All arrests
arrests.resample('M').sum().plot(figsize=(9,5))
plt.title('Number of All Arrests in Chicago, 2017-18',
          fontdict = {'fontsize': 14}, loc = 'left')

# Top 5 types of arrests
type_dum_arr.resample('M').sum().plot(figsize=(9,5)).\
             legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Number of Top 5 Types of Arrests in Chicago, 2017-18',
          fontdict = {'fontsize': 14}, loc = 'left')
plt.show()


# ### Neighborhood patterns 

# Load geojson of community areas
gdf = gpd.read_file("data/Boundaries - Community Areas (current).geojson")
gdf.set_index("area_num_1", inplace=True)

# Get counts of reported crimes by community area
num_rep_by_comm = crimes.groupby("community_area").size().to_frame()
num_rep_by_comm.columns = ["num_reported"]

# Get counts of arrests by community area
num_arr_by_comm = crimes[crimes.arrest == True].groupby("community_area").\
                  size().to_frame()
num_arr_by_comm.columns = ["num_arrested"]

# Merge counts to community area geodataframe
neighborhood = gdf.merge(num_rep_by_comm, left_index=True, right_index=True)
neighborhood = neighborhood.merge(num_arr_by_comm, left_index=True, 
                                  right_index=True)


# #### Community areas with the highest number of reported crimes:

nb_rep = neighborhood.sort_values('num_reported', ascending=False)[
         ['community', 'num_reported']][:20]
nb_rep.head()

# #### Community areas with the highest number of arrests:

nb_arr = neighborhood.sort_values('num_arrested', ascending=False)[
         ['community', 'num_arrested']][:20]
nb_arr.head()

# #### Chicago Community Areas with Top 20  Number of Reported Crimes, 2017-18:

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 8))

# Plot the total number of crimes reported
sns.set_color_codes("pastel")
sns.barplot(x="num_reported", y="community", data=nb_rep,
            label="All reported", color="b")

# Plot the crimes resulted in arrests
sns.set_color_codes("muted")
sns.barplot(x="num_arrested", y="community", data=nb_arr,
            label="Arrests", color="b")


ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="", xlabel="Number of crimes")
plt.show()


# #### Community Area Crime Heat Map, 2017-18:


sns.set(style="white")
neighborhood.plot(figsize = (10, 10), column='num_reported', cmap="OrRd", 
                  legend=True)
plt.title("Reported Crimes in Chicago, 2017-18", 
         fontdict={'fontsize': 16}, loc='left', pad=15)
plt.show()

neighborhood.plot(figsize = (10, 10), column='num_arrested', cmap="RdPu", 
                  legend=True)
plt.title("Arrests across in Chicago, 2017-18", 
          fontdict={'fontsize': 16}, loc='left', pad=15)
plt.show()


# ## 2.1 Data Augmentation

# #### Requesting block group data using Census ACS 5-Year Estimate (2017) API:

# Construct request URL for American Community Survey 5-Year Estimate (2017)
base_url = 'https://api.census.gov/data/2017/acs/acs5?get=NAME,'
varlist = ['B01003_001E', 'B19013_001E','B25003_001E', 'B25003_002E', 
           'B25003_003E']
base_url += (','.join(varlist))
base_url += '&for=block+group:*&in=state:17+county:031'

# Request and store ACS data
response = requests.get(base_url)
cen = pd.read_json(response.text)
header = ['name', 'population', 'median_income', 'total_tenure', 
          'owner_occupied', 'renter_occupied', 'state', 'county', 'tract', 
          'block_group']
cen = cen[1:]
cen.columns = header

# #### Match each crime report to block group GEOID using its (lat, long) coordinates and the geographical boundaries of Census block groups
#  
# Percentage of crime reports without coordinates
len(crimes[crimes['location'].isna()])/len(crimes)

# Drop crimes without coordinates and convert to geodataframe
gcrimes = crimes.dropna(subset=['location'])
gcrimes.location = gcrimes.location.apply(lambda x: shape(x))
gcrimes = gpd.GeoDataFrame(gcrimes, geometry='location')

# Read census block groups shapefile
blocks = gpd.read_file('data/censusblocks/cb_2016_17_bg_500k.shp')
blocks_to_join = blocks[['GEOID', 'geometry']]

# Get block group ID for crime report
blocks_to_join.crs = {'init': 'epsg:4326'}
gcrimes.crs = {'init': 'epsg:4326'}
gcrimes = gpd.sjoin(gcrimes, blocks_to_join, how="left", op='intersects')

# Create GEOID in census data from state, county, tract, and block group 
# numbers
cen['GEOID'] = cen.state + cen.county + cen.tract + cen.block_group

# Clean census data for merge
cen = cen.astype({'population': float, 'median_income': float, 
                  'total_tenure': float, 'owner_occupied': float, 
                  'renter_occupied': float})
cen['owner_occupancy_rate'] = cen.owner_occupied / cen.total_tenure
cen_to_merge = cen[['population', 'median_income', 'total_tenure', 
                    'owner_occupied', 'renter_occupied', 'GEOID', 
                    'owner_occupancy_rate']]

# Merge crimes and ACS variables
crimes_cen = gcrimes.merge(cen_to_merge, on='GEOID', how='left')
crimes_cen.head()


# ## 2.2 Block Characteristics

# ### i) Battery

bat = an.get_crime_type_data('BATTERY', crimes_cen)

# Blocks with most battery reports
bat_bg = an.get_top_blocks(bat)
bat_bg.head()

# Map blocks with most battery reports
an.map_blocks(bat_bg, blocks)

# Get summary stats on block group ACS variables and their correlation with 
# the number of batteries in the block
an.describe_blocks('BATTERY', bat_bg, cen_to_merge, 
                   outliers = ['170318391001'])


# ### ii) Homicide

hom = an.get_crime_type_data('HOMICIDE', crimes_cen)

# Blocks with most homicide reports
hom_bg = an.get_top_blocks(hom)
hom_bg.head()

# Map blocks with most battery reports
an.map_blocks(hom_bg, blocks)

# Get summary stats on block group ACS variables and their correlation with 
# the number of homicides in the block

an.describe_blocks("HOMICIDE", hom_bg, cen_to_merge)


# ### iii) Battery and Homicde in 2017 vs. 2018

# Battery 2017 sum stat
bat17 = an.get_crime_type_data('BATTERY', crimes_cen, year='2017')
bat_bg17 = an.get_top_blocks(bat17)
an.describe_blocks('BATTERY', bat_bg17, cen_to_merge, 
                   outliers=['170318391001'], plot=False)
# Battery 2018 sum stat
bat18 = an.get_crime_type_data('BATTERY', crimes_cen, year='2018')
bat_bg18 = an.get_top_blocks(bat18)
an.describe_blocks('BATTERY', bat_bg18, cen_to_merge, 
                   outliers=['170318391001'], plot=False)


# Homicide 2017 sum stat
hom17 = an.get_crime_type_data('HOMICIDE', crimes_cen, year='2017')
hom_bg17 = an.get_top_blocks(hom17)
an.describe_blocks('HOMICIDE', hom_bg17, cen_to_merge, plot=False)

# Homicide 2018 sum stat
hom18 = an.get_crime_type_data('HOMICIDE', crimes_cen, year='2018')
hom_bg18 = an.get_top_blocks(hom18)
an.describe_blocks('HOMICIDE', hom_bg18, cen_to_merge, plot=False)


# ### iv) Deceptive Practice vs. Sex Offense

# Deceptive Practice

dp = an.get_crime_type_data('DECEPTIVE PRACTICE', crimes_cen)
dp_bg = an.get_top_blocks(dp)
dp_bg.head()

# Get sum stat for number of deceptive practices at block level
dp_bg.describe()

an.map_blocks(dp_bg, blocks)
an.describe_blocks('DECEPTIVE PRACTICE', dp_bg, cen_to_merge, 
                   outliers = ['170318391001'])


# Sex offense

so = an.get_crime_type_data('SEX OFFENSE', crimes_cen)
so_bg = an.get_top_blocks(so)
so_bg.head()

# Get sum stat for number of sex offenses at block level
so_bg.describe()

an.map_blocks(so_bg, blocks)
an.describe_blocks('SEX OFFENSE', so_bg, cen_to_merge, 
                   outliers = ['170318391001'])


# ## 3.1 How did crime change in Chicago from 2017 to 2018
# 
# See notebook for writeup


# ## 3.2 Evaluation of Candidate's Statistics


# Get crimes data for 43rd Ward during the 28-day period in 2017 and in 2018
crimes18 = crimes[crimes.ward == '43']['2018-06-28':'2018-07-26']
crimes17 = crimes[crimes.ward == '43']['2017-06-28':'2017-07-26']

# Robberies

rob17 = an.get_crime_type_data('ROBBERY', crimes17)
rob18 = an.get_crime_type_data('ROBBERY', crimes18)
print('2018:', len(rob18))
print('2017:', len(rob17))
(len(rob18) - len(rob17)) / len(rob17)


# Batteries

bat17 = an.get_crime_type_data('BATTERY', crimes17)
bat18 = an.get_crime_type_data('BATTERY', crimes18)
print('2018:', len(bat18))
print('2017:', len(bat17))
(len(bat18) - len(bat17)) / len(bat17)


# Burglaries

bur17 = an.get_crime_type_data('BURGLARY', crimes17)
bur18 = an.get_crime_type_data('BURGLARY', crimes18)
print('2018:', len(bur18))
print('2017:', len(bur17))
(len(bur18) - len(bur17)) / len(bur17)


# Motor Vehicle theft

mvt17 = an.get_crime_type_data('MOTOR VEHICLE THEFT', crimes17)
mvt18 = an.get_crime_type_data('MOTOR VEHICLE THEFT', crimes18)
print('2018:', len(mvt18))
print('2017:', len(mvt17))
(len(mvt18) - len(mvt17)) / len(mvt17)


# All crimes

(len(crimes18) - len(crimes17)) / len(crimes17) 


# Percentage change of all crimes data for 43rd Ward in 2017 and 2018
crimes18_tot = crimes[crimes.ward == '43']['2018-01-01':'2018-12-31']
crimes17_tot = crimes[crimes.ward == '43']['2017-01-01':'2017-12-31']
(len(crimes18_tot) - len(crimes17_tot)) / len(crimes17_tot) 
