#!/usr/bin/env python
# coding: utf-8

# # Project 1: Exploratory Data Analysis

# For this project, the 'House_Prices.csv' dataset will be analyzed. The data in this dataset is related to house sale prices for a certain region of the United States and includes homes sold between May 2014 and May 2015. 

# ### Importing the modules

# The Exploratory Data Analysis for this project will include the following modules:

# In[1]:


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.formula.api import ols
from pathlib import Path


# ### Importing the dataset

# The dataset will be imported into a Pandas DataFrame.

# In[2]:


# importing the dataset
df = pd.read_csv('C:/Users/19145/Documents/CS675_Jupyter_Notebooks/project_files/House_Prices.csv', index_col=0) # -- This will allow the selection of data by labels.

df.shape


# The output above is the size of the dataset. According to the output, this dataset consists of 21613 rows (records) and 20 columns (features).
# This is a rather large dataset so it will be subsampled so that it will be easier to work with in jupyter.
# 
# The sampled dataframe will be saved to a csv file so that the same sample is used when the code is executed on different machines and to maintain consistency in the data reported.

# In[3]:


# This code is only executed once so that the same sample is used.
#'''
#df = df.sample(n=4000)

# Save the sampled dataset. This is only performed once.
#filepath = Path('C:/Users/19145/Documents/CS675_Jupyter_Notebooks/project_files/house_prices_sampled.csv')  
#filepath.parent.mkdir(parents=True, exist_ok=True)  
#df.to_csv(filepath, index=False)
#'''


# Now that the dataset has been subsampled, the first couple of records will be observed as a glimpse into the data that will be explored.

# In[4]:


# Import the sampled dataset
df = pd.read_csv('C:/Users/19145/Documents/CS675_Jupyter_Notebooks/project_files/house_prices_sampled.csv')
#df.set_index('id')

# Display the first 10 rows of the dataset
df.head(10)


# From this sample of the dataset, it can be observed that each record is uniquely identified by the 'id' and 'date' columns. Another observation is the type of features in the dataset. Features can either be continuous, discrete, or categorical.
# 
# An example of a continuous feature of this dataset would be the home price.
# 
# An example of a discrete feature of this dataset would be the number of bedrooms.
# 
# An example of a categorical feature of this dataset would be waterfront ('0' meaning 'no waterfront' and '1' meaning there is a waterfront).
# 
# The dataset can be further summarized by obtaining the data types for each feature.

# In[5]:


# Return the data type of each feature.
df.dtypes


# ### Counting for missing values and checking for duplicates

# To begin the Exploratory Data Anaysis, we count the number of missing values for each feature in the dataset (if any).

# In[6]:


# Count the number of missing values for each feature.
df.isna().sum()


# From this observation, there are no missing values in our dataset.
# We will also check for any duplicates in the dataset.

# In[7]:


# Find duplicate rows across all the columns
df[df.duplicated()]


# This dataset contains no missing values or duplicates, confirming a clean dataset.

# ### Summarizing the dataset

# Now, we obtain a summary of the dataset in the form of descriptive statistics.

# In[8]:


# Get a summary of the data in the dataset.
df.describe()


# The above summary gives an overview of the features of this dataset. Some of the helpful statistics from this summary include the range of house prices, the average number of bedrooms and floors, and the condition of the majority of homes that were on the market.

# ### Plotting the data

# A few plots will be designed to represent a specific analysis of the dataset.
# For time series plots, a new dataframe will be created that features a formatted version of the date field from the original dataset.
# The date format will be chaged to a 'YYYY-MM-DD' format.

# In[9]:


# Create a new dataframe using the original dataset, then change the format of the date values.
df_new_date_format = df
df_new_date_format['date'] = pd.to_datetime(df_new_date_format['date'], format="%Y/%m/%d")

df_new_date_format.head(10)


# As we can see, the date column is in a new format that can be used for constructing the visualizations.
# We then sort this new dataset by date, then confirm by displaying the first couple of entries.

# In[10]:


# Create a new dataframe for the sorted dataset
df_by_date = df_new_date_format.sort_values(by='date')

df_by_date.head(10)


# In[11]:


# Create a new column for storing the year-month of each sale.
df_by_date['year_month'] = df_by_date['date'].dt.to_period('M')
df_by_date[['year_month','price']]


# ### Trending Sales

# The following visualization will showcase the trend of sales over time. The mean and standard deviation of home prices will also be plotted to identify where the majority of values exist.

# In[12]:


# Calculate the mean and std of all sales
df_by_date['price_avg'] = df_by_date['price'].mean()
avg_sale = df_by_date.price_avg

std_sale = df_by_date['price'].std()
df_by_date['price_abv_std'] = avg_sale + std_sale #--sales above the standard deviation
df_by_date['price_blw_std'] = avg_sale - std_sale #--sales below the standard deviation

std_sale_abv = df_by_date.price_abv_std
std_sale_blw = df_by_date.price_blw_std

# Get the x-and-y values to be used in the plot
date = df_by_date.date
price = df_by_date.price
    
# Create a plot for observing the sale of homes over time along with the mean and standard deviations
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Housing Sales 2014-15')
plt.plot(date, price)
plt.plot(date, avg_sale, color='yellow')
plt.plot(date, std_sale_abv, color='green')
plt.plot(date, std_sale_blw, color='green')
plt.show()


# From the graph above, there appears to be a consistent trend in house prices. As expected, the majority of the homes sold stayed within the standard deviation of the average. There doesn't appear to be any seasonality when it comes to the price of homes sold.
# 
# Below, we will look at a specified period to dive a little deeper into the trend.

# In[13]:


# Create a dataframe for sales that took place from July 2014 through September 2014
df_three_months = df_by_date[(date >= '2014-07-01') & (date <= '2014-09-30')]

# Create a plot for observing the sale of homes during this time period
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Housing Sales from 2014-07-01 to 2014-09-30')
plt.plot(df_three_months.date, df_three_months.price_avg)
plt.plot(df_three_months.date, df_three_months.price_abv_std)
plt.plot(df_three_months.date, df_three_months.price_blw_std)
plt.scatter(df_three_months.date, df_three_months.price, color='purple')
plt.show()


# The chart above displays the sale of homes from July 2014 through September 2014. As we can see, there are a few outliers in this trend. The outliers represent homes that were highly expensive. Despite these couple of outliers, the prices of homes sold remained consistent, with the majority of them falling within the range of the average.

# ### Trending Living Space

# The next trend that will be examined is the living space of the homes sold.

# In[14]:


# Calculate the mean and std of living space
df_by_date['living_avg'] = df_by_date['sqft_living'].mean()
avg_living_sp = df_by_date.living_avg

std_living = df_by_date['sqft_living'].std()
df_by_date['living_abv_std'] = avg_living_sp + std_living #--living space above the standard deviation
df_by_date['living_blw_std'] = avg_living_sp - std_living #--living space below the standard deviation

std_living_abv = df_by_date.living_abv_std
std_living_blw = df_by_date.living_blw_std

# Get the x-and-y values to be used in the plot
date = df_by_date.date
living = df_by_date.sqft_living
    
# Create a plot for observing the living space of homes sold over time along with the mean and standard deviations
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Living Space')
plt.title('Living Space of House Sales 2014-15')
plt.plot(date, living)
plt.plot(date, avg_living_sp, color='yellow')
plt.plot(date, std_living_abv, color='green')
plt.plot(date, std_living_blw, color='green')
plt.show()


# From the graph above, it can been seen that there is a consistent trend when it comes to the living space of homes sold. There doesn't seem to be any seasonality to report from this data.

# In[15]:


# Create a dataframe for the living space of homes sold between July 2014 and November 2014
df_four_months_living = df_by_date[(date >= '2014-07-01') & (date <= '2014-10-31')]

# Create a plot for observing the living space of homes sold during this time period
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Living Space')
plt.title('Living Space of Homes Sold from 2014-07-01 to 2014-10-31')
plt.plot(df_four_months_living.date, df_four_months_living.living_avg)
plt.plot(df_four_months_living.date, df_four_months_living.living_abv_std)
plt.plot(df_four_months_living.date, df_four_months_living.living_blw_std)
plt.scatter(df_four_months_living.date, df_four_months_living.sqft_living, color='purple')
plt.show()


# The trend here shows that there is still some consistency in the living space of homes. The outliers appear to be close to the upper boundary of the standard deviation. There are a handful of outliers that are far beyond that boundary. This could represent the homes that were sold at the highest prices.

# ### Trending Lot (Land) Space

# In[16]:


# Calculate the mean and std of land space
df_by_date['lot_avg'] = df_by_date['sqft_lot'].mean()
avg_lot_sp = df_by_date.lot_avg

std_lot = df_by_date['sqft_lot'].std()
df_by_date['lot_abv_std'] = avg_lot_sp + std_lot #--lot space above the standard deviation
df_by_date['lot_blw_std'] = avg_lot_sp - std_living #--lot space below the standard deviation

std_lot_abv = df_by_date.lot_abv_std
std_lot_blw = df_by_date.lot_blw_std

# Get the x-and-y values to be used in the plot
date = df_by_date.date
land = df_by_date.sqft_lot
    
# Create a plot for observing the land space of homes sold over time along with the mean and standard deviations
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Land (Lot) Space')
plt.title('Land (Lot) Space of House Sales 2014-15')
plt.plot(date, land)
plt.plot(date, avg_lot_sp, color='yellow')
plt.plot(date, std_lot_abv, color='green')
plt.plot(date, std_lot_blw, color='green')
plt.show()


# From this graph, there doesn't appear to be any seasonality even though their are plenty of spiking in the trend. The highest spikes seem to occur somewhere between September of 2014 and February of 2015. The next plot will investigate this period of time.

# In[17]:


# Create a dataframe for the lot space of homes sold between September 2014 and February 2015
df_five_months_lot = df_by_date[(date >= '2014-09-01') & (date <= '2015-01-31')]

# Create a plot for observing the lot space of homes sold during this time period
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Lot Space')
plt.title('Lot Space of Homes Sold from 2014-09-01 to 2015-01-31')
plt.plot(df_five_months_lot.date, df_five_months_lot.lot_avg)
plt.plot(df_five_months_lot.date, df_five_months_lot.lot_abv_std)
plt.plot(df_five_months_lot.date, df_five_months_lot.lot_blw_std)
plt.scatter(df_five_months_lot.date, df_five_months_lot.sqft_lot, color='purple')
plt.show()


# From this graph, we can see the spikes in the plot of trending lot space was the result of these few outliers.

# ### Distribution of Continuous Features

# The following graphs will show the distribution of sales, living space, and lot space to identify variance within the dataset.

# In[18]:


# Distribution plot for price
sns.displot(df['price'], kind='kde')
# Distribution plot for living space
sns.displot(df['sqft_living'], kind='kde', color='green')
# Distribution plot for land space
sns.displot(df_by_date['sqft_lot'], kind='kde')


# From these distribution plots, it can seen that the data skews in a specific direction. This is especially evident with the distribution of lot space.
# 
# For sale price, the distribution tells us that any home that is sold on the market will be priced somewhere between 100,000 and 1,000,000.
# 
# For living space, the distribution tells us that the living space of a home sold will fall between 0 and 5000 sq. ft.
# 
# For lot space, the distibution tells us that the lot space of a home sold will be less than 100,000 sq. ft.
# 
# The feature with the highest variance appears to be the living space, while the lot space has the lower variance.

# ### Trend of homes based on condition

# The trend of homes sold will now be investigated.

# In[19]:


# Get the x-and-y values to be used in the plot
date = df_by_date.date
cond = df_by_date.condition
    
# Create a plot for condition of homes sold
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Condition')
plt.title('Condition of homes sold 2014-15')
plt.scatter(date, cond)


# The trend clearly shows that the majority of homes sold were either in average or above average condition.
# 
# Next, we will look at the condition of homes based on the year they were built.

# In[20]:


# Get the x-and-y values to be used in the plot
yr = df_by_date.yr_built
cond = df_by_date.condition
    
# Create a plot for condition of homes sold
plt.figure(figsize=(12,8))
plt.xlabel('Year Built')
plt.ylabel('Condition')
plt.title('Condition of homes based on the Year Built')
plt.scatter(yr, cond)


# The graphs shows that the year the home was built has no relationship to the condition since the majority of the homes sold were in average condition or better.

# ### Condition, Grade, and View

# The following visualizations will show the distribution of some of the discrete features of the dataset.
# 
# We'll start by looking at the trend of condition over time.

# In[21]:


# Get the x-and-y values to be used in the plot
date = df_by_date.date
cond = df_by_date.condition
    
# Create a plot for condition of homes sold
plt.figure(figsize=(12,8))
plt.xlabel('Date')
plt.ylabel('Condition')
plt.title('Condition of homes sold 2014-15')
plt.scatter(date, cond)


# As we can see, the date of the sale doesn't have any relationship to the sale of the home. The majority of homes sold were either in average or above average condition.
# 
# Let's look at what happens when condition is plotted against the year it was built.

# In[22]:


# Get the x-and-y values to be used in the plot
yr = df_by_date.yr_built
cond = df_by_date.condition
    
# Create a plot for condition of homes sold
plt.figure(figsize=(12,8))
plt.xlabel('Year Built')
plt.ylabel('Condition')
plt.title('Condition of homes based on year built')
plt.scatter(yr, cond)


# Here, it shows that the year a home was originally built has no relation to the condition of the home. In other words, when a home was originally built does not tell us anything about its condition.

# Let's look at the grade of the home based on the year it was built.

# In[23]:


# Get the x-and-y values to be used in the plot
grade = df_by_date.grade
yr = df_by_date.yr_built
    
# Create a plot for condition of homes sold
plt.figure(figsize=(12,8))
plt.xlabel('Year')
plt.ylabel('Grade')
plt.title('Grade of homes based on the year built')
plt.scatter(yr,grade)


# The graph above shows that the condition of homes built have improved over time. This is evident by the increase of homes with a grade of 11 of higher after 1980.
# 
# Let's look at this further.

# In[24]:


# Create a new dataframe for filtering data for homes built after 1980
df_after1980 = df_by_date[(df_by_date.yr_built >= 1980)]

# Get the x-and-y values to be used in the plot
yr = df_after1980.yr_built
grade = df_after1980.grade

    
# Create a plot for the grade of homes built after 1980
plt.figure(figsize=(12,8))
plt.xlabel('Year')
plt.ylabel('Grade')
plt.title('Grade of homes built after 1980')
plt.scatter(yr,grade)


# This graph shows that homes built after 1984 were a grade of 11 of higher. Let's contrast this with the grade of homes built prior to 1980.

# In[25]:


# Create a new dataframe for filtering data for homes built before 1980
df_before1980 = df_by_date[(df_by_date.yr_built < 1980)]

# Get the x-and-y values to be used in the plot
yr = df_before1980.yr_built
grade = df_before1980.grade

    
# Create a plot for the grade of homes built before 1980
plt.figure(figsize=(12,8))
plt.xlabel('Year')
plt.ylabel('Grade')
plt.title('Grade of homes built before 1980')
plt.scatter(yr,grade)


# From this graph, it shows that the number of homes built prior to 1980 with a grade of 11 or higher were very sparse. What is also noticable is the number of homes with a grade lower than 6. This confirms our theory that the construction of homes had improved in this area over time.

# ### Distribution of Discrete Features

# The graphs below will show the distribution of the discrete features of condition, grade, and view.

# In[26]:


# Distribution plot for price
sns.displot(df_by_date['condition'])
# Distribution plot for living space
sns.displot(df_by_date['grade'], color='green')
# Distribution plot for land space
sns.displot(df_by_date['view'], color='purple')


# The above distribution plots show that there isn't a lot of variance when it comes to the condition and view features. There is plenty of variance with the grade feature, however. Much of that data is skewed to the right. We can see that most of the homes in the dataset have a grade of 7 or 8, which could be consider average.

# ### Correlation of Features

# The following correlation matrix will provide analysis of the relationship between all the features in the dataset.

# In[27]:


'''
Before we create the correlation matrix, we will have to drop a few columns that were created when 
designing the time series graphs.

'''
df_corr = df_by_date.drop(['price_avg', 'price_abv_std', 'price_blw_std', 'living_avg', 'living_abv_std', 'living_blw_std', 'lot_avg', 'lot_abv_std', 'lot_blw_std'], axis=1)

# Create the correlation matrix and round values to 3 decimals
df_corr.corr().round(3)


# To provide a clearer picture of the correlation between the features of this dataset, a colormap will be applied.

# In[28]:


colorful_corr = df_corr.corr().round(3)
colorful_corr.style.background_gradient(cmap='coolwarm')


# This colorful version of the correlation matrix allows us to identify what features share a strong relationship with each other. From this matrix, it can be seen that the living space (sqft_living) feature has a strong positive relationship with the space above (sqft_above) feature. It also has a strong positive relatoinship with the number of bathrooms.
# 
# Another interesting relationship is the one between bathrooms and grade. Grade refers to the quality of construction and design of a home. The positive correlation between these two features suggest that homes that are better constructed have more bathrooms.
# 
# When it comes to the target feature, the price of the sale, living space (sqft_living), construction grade (grade), space above the building (sqft_above), and the living space of the nearest 15 neighbors (sqft_living15) share the strongest correlations, all of them positive. This needs to be considered when it comes to constructing a regression model for predicting future housing prices.

# ### Example: Correlation between bathrooms and grade

# The scatterplot below will show the correlation between the bathrooms and grade features.

# In[29]:


# Create a scatterplot for visualizing the correlation between the grade and bathroom features
sns.scatterplot(data=df_by_date, x=df_by_date['grade'], y=df_by_date['bathrooms'])
plt.show()


# The barplot confirms the positive correlation between the number of bathrooms and the grade of the home sold. It also shows that there is a high variance in the number of bathrooms for homes with a grade of 7.

# ### Examining Outliers

# The visualizations below will help provide more analysis on the outliers from the dataset. These visualizations focus on the continuous features price, living space, and lot space.

# In[30]:


# Create a boxplot for the price feature
boxplot = df_by_date.boxplot(column='price', by='year_month', figsize=(12, 14))


# The boxplot above provides an analysis of the price of homes that were sold each month. The variance for each month appears to be identical in terms of their skewness. November appears to be the one month where the median price is exactly in the middle. The outliers for each month appear to be clustered together, with a few data points extending beyond the maximum value relative to the boxes.  

# In[31]:


# Create a boxplot for the living space
boxplot = df_by_date.boxplot(column='sqft_living', by='year_month', figsize=(12, 14))


# The boxplot above shows that there aren't that many outliers for the sqft_living data as they are for the price data. They're a bit more scattered. However, the whiskers indicate a wider spread of data for this feature. This was already shown in the distribution graphs.

# In[32]:


# Create a boxplot for the lot space
boxplot = df_by_date.boxplot(column='sqft_lot', by='year_month', figsize=(12, 14))


# The boxplot for sqft_lot show that there is poor variance for this feature. This is probably the reason for the presence of so many outliers, many of thenm tightly clustered togehter.

# ### Performing ANOVA

# The ANOVA test will be performed to examine the effects of two features on our target feature, which is the price.
# The features that will be used in this test is the 'sqft_living' and 'sqft_lot' features.
# Before we can perform the test, a new column needs to be created for categorizing both the 'sqft_living' and 'sqft_lot' features so they can be used in the ANOVA formula.

# In[33]:


# create a list of conditions for the 'sqft_living' column
living_cat  = [ 
    (df_by_date['sqft_living'] <= 1500),
    (df_by_date['sqft_living'] > 1500) & (df_by_date['sqft_living'] <= 3000),
    (df_by_date['sqft_living'] > 3000)
    ]

# create a list of the values to assign to the new column
values = ['small', 'medium', 'large']

# create a new column in the dataframe
df_by_date['sqft_living_cat'] = np.select(living_cat, values)

# create a list of conditions for the 'sqft_lot' column
lot_cat  = [ 
    (df_by_date['sqft_lot'] <= 10000),
    (df_by_date['sqft_lot'] > 10000) & (df_by_date['sqft_lot'] <= 20000),
    (df_by_date['sqft_lot'] > 20000)
    ]

# create a list of the values to assign to the new column
values_02 = ['small', 'medium', 'large']

# create a new column in the dataframe
df_by_date['sqft_lot_cat'] = np.select(lot_cat, values_02)


# Now we execute the ANOVA formula.

# In[34]:


#perform two-way ANOVA
model = ols('price ~ C(sqft_living_cat) + C(sqft_lot_cat) + C(sqft_living_cat):C(sqft_lot_cat)', data=df_by_date).fit()
sm.stats.anova_lm(model, typ=2)


# ### Interpreting the Results

# Based on the results from the ANOVA formula:
# 
# sqft_living_cat p-value: 0
# 
# sqft_lot_cat p-value: ~ 0.01
# 
# sqft_living_cat:sqft_lot_cat: ~ 0.01 
# 
# The p-values of sqft_living_cat and sqft_lot_cat tell us that both the living space and lot space have a significant effect on the price of a home that is sold on the market. The p-value for the interaction effect tells us that there is no signficant interaction effect between the two features.

# ### Visualization of the Results

# In[35]:


df_by_date.boxplot('price', by='sqft_living_cat')
df_by_date.boxplot('price', by='sqft_lot_cat')


# The boxplots above provide an analysis of the price of each home based on the categories of both sqft_living and sqft_lot.
# 
# For sqft_living, there is wider variance of data for homes with a large living space. the data for this group also appears to be slightly more skewed in the positive direction than for homes with a medium or small living space.
# 
# For sqft_lot, the variance appears to be the same for all three categories. The median for homes with a medium lot space appears to be slightly lower than the others, resulting in a slight positive skew.
# 
# For both features, the outliers are all close together and just touch the upper whisker of the box plots.

# In[ ]:




