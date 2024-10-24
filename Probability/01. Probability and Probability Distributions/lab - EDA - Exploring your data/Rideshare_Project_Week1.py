#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - Understanding Your Dataset
# 
# Welcome to the second notebook of the exploratory data analysis (EDA) series, where you will get your hands dirty applying the skills you have learned in the course on an actual data problem, similar to those you might encouter in real life! This is a part of a series, which contains five notebooks, each of them placed on different weeks of this course. There is very little mathematics instruction in these notebooks, but rather practical implementations of the concepts you learned using various python libraries.
# 
# For this notebook you will use the data on ridesharing in the year 2022 in the city of Chicago, which can be found [here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p/data).
# 
# We have already downloaded the dataset for you and put it in the folder together with this notebook. If you check the link above, you might notice that the dataset includes hundreds of millions of rows. This translates to tens of gigabytes and is too large for working in this environment. That's why the dataset has been preprocessed to include only the data from 2022 and downsampled by a factor of 100 to easily fit into the environment and make your experience more pleasant.
# 
# In this notebook you will mostly use the Pandas library. If you are not familiar with it, you can check out the Pandas tutorial notebook.
# 
# ### Learning Objectives:
# In this notebook you will use the following concepts from the course in a practical setting:
#  - Probability
#  - Conditional probability
#  - Distributions
# 

# # 1. Import the Python Libraries
# 
# As usual, the first thing you need to do is import the libraries that you will use in this notebook. `pandas` will help you load and manipulate data, while `matplotlib` will be used for plottting.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# # 2. Load the Dataset
# 
# The next step is to load the dataset. The dataset has been downsampled by a factor of 100 to work smoothly in this environment.

# In[2]:


# Open the dataset
df = pd.read_csv("data/rideshare_2022.csv", parse_dates=['Trip Start Timestamp', 'Trip End Timestamp'])

# Show the first five rows of the dataset
df.head()


# # 3 Explore the Dataset
# 
# In the cell above, you have opened the dataset and displayed the first five rows. Have a closer look at the output of the cell above. The dataset consists of the following columns:
# 
# - `Trip ID`: A unique identifier for the trip.
# 
# 
# - `Trip Start Timestamp`: When the trip started, rounded to the nearest 15 minutes.
# 
# 
# - `Trip End Timestamp`: When the trip ended, rounded to the nearest 15 minutes.
# 
# 
# - `Trip Seconds`: Time of the trip in seconds.
# 
# 
# - `Trip Miles`: Distance of the trip in miles.
# 
# 
# - `Pickup Census Tract`: The Census Tract where the trip began. This column often will be blank for locations outside Chicago.
# 
# 
# - `Dropoff Census Tract`: The Census Tract where the trip ended. This column often will be blank for locations outside Chicago.
# 
# 
# - `Pickup Community Area`: The Community Area where the trip began. This column will be blank for locations outside Chicago.
# 
# 
# - `Dropoff Community Area`: The Community Area where the trip ended. This column will be blank for locations outside Chicago.
# 
# 
# - `Fare`: The fare for the trip, rounded to the nearest $2.50. 
# 
# 
# - `Tip`: The tip for the trip, rounded to the nearest $1.00. Cash tips will not be recorded.
# 
# 
# - `Additional Charges`: The taxes, fees, and any other charges for the trip.
# 
# 
# - `Trip Total`: Total cost of the trip. This is calculated as the total of the previous columns, including rounding.
# 
# 
# - `Shared Trip Authorized`: Whether the customer agreed to a shared trip with another customer, regardless of whether the customer was actually matched for a shared trip.
# 
# 
# - `Trips Pooled`: If customers were matched for a shared trip, how many trips, including this one, were pooled. All customer trips from the time the vehicle was empty until it was empty again contribute to this count, even if some customers were never present in the vehicle at the same time. Each trip making up the overall shared trip will have a separate record in this dataset, with the same value in this column.
# 
# 
# - `Pickup Centroid Latitude`: The latitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# - `Pickup Centroid Longitude`: The longitude of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# - `Pickup Centroid Location`: The location of the center of the pickup census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# - `Dropoff Centroid Latitude`: The latitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# - `Dropoff Centroid Longitude`: The longitude of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# - `Dropoff Centroid Location`: The location of the center of the dropoff census tract or the community area if the census tract has been hidden for privacy. This column often will be blank for locations outside Chicago.
# 
# 
# Run the cell below to print out the column names and inspect the number of non-null values and the data type of each column. 

# In[3]:


df.info()


# ## 3.1 Select columns of interest
# 
# At this point, you have seen what the dataset looks like. Take a moment to think of your next steps. Which columns would you explore further? Is there a column that has a problematic number of null values? Are there any columns that you are not interested in?
# 
# For exploratory data analysis it is perfectly fine to select only the columns that you are interested in and drop the remainder. This will not only make your dataframe easier to work with, but also reduce its size, making your operations faster.
# 
# In the cell below you will select a subset of the columns, which are the ones you will be interested in for this notebook. If you keep only the columns that are pre-selected in the cell below, you will reduce the file size by about a half. This can make a difference of whether you can fit the file into the memory or not, especially with larger files. This code will also rename the columns to remove white spaces.

# In[4]:


columns_of_interest = ['Trip Start Timestamp', 'Trip Seconds',
       'Trip Miles', 'Fare', 'Tip', 'Additional Charges', 'Trip Total', 'Shared Trip Authorized',
       'Trips Pooled', 'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 'Dropoff Centroid Latitude',
       'Dropoff Centroid Longitude']

df = df[columns_of_interest]

# Rename all the columns to not include whitespace
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})

# Check the info on the cleaned-up dataset
df.info()


# # 4. Visualize the data
# 
# To understand the data better, it often makes sense to visualize it. This helps you understand how the data is distributed. You can start by plotting the number of rides in a given day. For this it would be useful to have another column that just contains the date. The code in the cell below will create a new column which takes the `trip_start_timestamp` and converts it into a date.

# In[5]:


df['date'] = pd.to_datetime(df['trip_start_timestamp'].dt.date)

df.head()


# In[6]:


# Select the column which you want to plot.
column_to_plot = 'date'

# Plot the histogram of the desired column
df.hist(column_to_plot, density=True)


# What you have plotted above is the distribution of the rides throughout the year. Note the code above also set the `dentsity=True`. This is so that the histogram is scaled to look like a probability density function like the ones you saw on Lesson 2. This means scaling the plot so that the area of the bars equals 1. What does this distribution look like to you? Is it similar to any of the distributions you saw in the videos? 
# 
# Although the distribution is slightly smaller for earlier dates, you could probably say that the rides are quite uniformly distributed throughout the year. Just note that this is not the actual distribution of the dates of cab rides, but rather an estimate based on the observations you have. Since this is real-world data, there are some fluctuations.
# 
# Now change the `column_to_plot` variable above to some other column name to observe the distributions of other variables. Some interesting ones might be `fare`, `tip` or `trip_length`. These variables can tell you how far drivers have to drive and how much they are getting paid for it.
# 
# Lets look together at the `tip` column.

# In[7]:


# Select the column which you want to plot.
column_to_plot = 'tip'

# Plot the histogram of the desired column
df.hist(column_to_plot, density=True, bins = 100);


# What can you say about the distribution of tips? This one looks a bit weird, right? What could explain this strange distribution? What do you think the large bar on the left corresponds to?
# 
# What is actually happening here is that the majority of the people do not tip, and that's why you see a large bar at tip = 0. 
# 
# Based on the data, you can calculate the probability of the customer tipping. You can do this by simply calculating the proportion of customers that actually tipped from the total number of rides.

# In[8]:


# Create a boolean series that distinguishes between tippers and no-tippers
tippers = df['tip'] > 0
# Count the number of tippers
number_of_tippers = tippers.sum()
# Count the total number of rides
total_rides = len(df)

# Calculate the fraction of people who tip
fraction_of_tippers = number_of_tippers / total_rides
print(f'The percentage of riders who tip is {fraction_of_tippers*100:.0f}%.')


# In the next cell you will create a new dataframe, where you will remove the non-tippers (the ones who gave a tip of zero). Then you can replot the histogram and see how it looks without the large bar at tip = 0.

# In[9]:


# Create a dataframe That only consists of tippers (conditioned on the boolean series)
df_tippers = df[tippers]

# Now re-plot the above histogram, but only for tippers
df_tippers.hist('tip', density=True, bins = 100);


# You can see now that the distribution got a much more interesting shape. What you are actually doing here is conditioning the original variable `tip`. You are ploting the distribution of tips given that a tip was actually given, or given that `tip>0` if you want it in mathematical terms. In other words, you are discarding part of your data, where `tip=0`, and finding the distribution of the remaining data.
# 
# # 5. Split the Data Into Interesting Subsets
# 
# The next thing you can check is if you can create any subsets of data and have a look at conditional distributions over these subsets. For example, you might be interested, to know whether there are more rides on the weekend than during the week, or if people tip more on weekends. This can help you figure out whether there are any differences in demand during the week and helps you adjust the supply of drivers.
# 
# For this you will first create a new column called `weekday`, where you will store the information on the day of the week.

# In[10]:


# Extracting the day of the week is simple when you have it in datetime format.
df['weekday'] = df["date"].dt.day_name()

df.head()


# Now you can count the number of riders on a given day of the week.

# In[11]:


# Count the number of rides each day
daily_ride_counts = df['weekday'].value_counts()

# List of weekdays. You will use it to reorder the counts, as they are in random order.
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Reorder the series given weekdays
daily_ride_counts = daily_ride_counts.reindex(WEEKDAYS)

daily_ride_counts


# And in the same manner, you will calculate the number of tippers on a given day of the week.

# In[12]:


df_tippers = df[df['tip'] > 0]
# Count the number of tips given each day
daily_tippers_counts = df_tippers['weekday'].value_counts()

# Reorder the series given weekdays
daily_tippers_counts = daily_tippers_counts.reindex(WEEKDAYS)

daily_tippers_counts


# Now you can calculate the percentage of customers tipping on each day of the week.

# In[13]:


df_daily_aggregation = pd.concat([daily_ride_counts, daily_tippers_counts], axis=1, keys=['ride_count', 'tippers_count'])
df_daily_aggregation["tips_percentage"] = df_daily_aggregation['tippers_count'] / df_daily_aggregation['ride_count'] * 100

df_daily_aggregation


# What you have just calculated are conditional probabilities: What is the probability of someone tipping, given a certain day of the week? Or if you write it with an equation: $ P(tip|weekday) $.
# Now you can have another look at the numbers and see if there are some important insights!
# 
# You can see that there are significantly more rides on Fridays and Saturdays than on the other days of the week, however the percentage of the tippers does not change much.
# 
# You can use the cell below to save your modified dataframe. You dont need to do that, as the dataframe for the next lab is already provided.

# In[15]:


# Uncomment the line below if you want to save your dataframe.
df.to_csv("data/rideshare_2022_user.csv", index=False)


# **Congratulations on finishing this lab.** You have used the implementation of quite a few concepts covered in this course: probabilities, distributions and conditional probabilities. On top of that you have practiced Pandas a little bit. If you liked this exercise, look out for another similar notebook next week!

# In[ ]:




