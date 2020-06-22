#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../Pierian_Data_Logo.PNG"></a>
# <strong><center>Copyright by Pierian Data Inc.</center></strong> 
# <strong><center>Created by Jose Marcial Portilla.</center></strong>

# # Keras API Project Exercise - Solutions
# 
# ## The Data
# 
# We will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# ## NOTE: Do not download the full zip from the link! We provide a special version of this file that has some extra feature engineering for you to do. You won't be able to follow along with the original file!
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Our Goal
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can we build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. Keep in mind classification metrics when evaluating the performance of your model!
# 
# The "loan_status" column contains our label.
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# ## Starter Code
# 
# #### Note: We also provide feature information on the data as a .csv file for easy lookup throughout the notebook:

# In[1]:


import pandas as pd
import time
import random
# In[2]:


data_info = pd.read_csv('../input/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[4]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('../input/lending_club_loan_two.csv')


# In[8]:


df.info()


# # Project Tasks
# 
# **Complete the tasks below! Keep in mind is usually more than one way to complete the task! Enjoy**
# 
# -----
# ------
# 
# # Section 1: Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**
# 
# 
# ----

# **TASK: Since we will be attempting to predict loan_status, create a countplot as shown below.**

# In[9]:


# CODE HERE


# In[10]:


sns.countplot(x='loan_status',data=df)


# **TASK: Create a histogram of the loan_amnt column.**

# In[11]:


# CODE HERE


# In[12]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


# **TASK: Let's explore correlation between the continuous feature variables. Calculate the correlation between all continuous numeric variables using .corr() method.**

# In[13]:


# CODE HERE


# In[14]:


df.corr()


# **TASK: Visualize this using a heatmap. Depending on your version of matplotlib, you may need to manually adjust the heatmap.**
# 
# * [Heatmap info](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)
# * [Help with resizing](https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot)

# In[15]:


# CODE HERE


# In[16]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# **TASK: You should have noticed almost perfect correlation with the "installment" feature. Explore this feature further. Print out their descriptions and perform a scatterplot between them. Does this relationship make sense to you? Do you think there is duplicate information here?**

# In[17]:


# CODE HERE


# In[18]:


feat_info('installment')


# In[19]:


feat_info('loan_amnt')


# In[20]:


sns.scatterplot(x='installment',y='loan_amnt',data=df,)


# **TASK: Create a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[21]:


# CODE HERE


# In[22]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# **TASK: Calculate the summary statistics for the loan amount, grouped by the loan_status.**

# In[23]:


# CODE HERE


# In[24]:


df.groupby('loan_status')['loan_amnt'].describe()


# **TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. What are the unique possible grades and subgrades?**

# In[25]:


# CODE HERE


# In[26]:


sorted(df['grade'].unique())


# In[27]:


sorted(df['sub_grade'].unique())


# **TASK: Create a countplot per grade. Set the hue to the loan_status label.**

# In[28]:


sns.countplot(x='grade',data=df,hue='loan_status')


# **TASK: Display a count plot per subgrade. You may need to resize for this plot and reorder the x axis. Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status**

# In[29]:


#CODE HERE


# In[30]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )


# In[31]:


# CODE HERE


# In[32]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# **TASK: It looks like F and G subgrades don't get paid back that often. Isloate those and recreate the countplot just for those subgrades.**

# In[33]:


# CODE HERE


# In[34]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# **TASK: Create a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[35]:


# CODE HERE


# In[ ]:


# get_ipython().run_line_magic('debug', '')
df['loan_status'].unique()


# In[37]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[38]:


df[['loan_repaid','loan_status']]


# **CHALLENGE TASK: (Note this is hard, but can be done in one line!) Create a bar plot showing the correlation of the numeric features to the new loan_repaid column. [Helpful Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html)**

# In[39]:


#CODE HERE


# In[40]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# ---
# ---
# # Section 2: Data PreProcessing
# 
# **Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**
# 
# 

# In[ ]:


df.head()


# # Missing Data
# 
# **Let's explore this missing data columns. We use a variety of factors to decide whether or not they would be useful, to see if we should keep, discard, or fill in the missing data.**

# **TASK: What is the length of the dataframe?**

# In[ ]:


# CODE HERE


# In[ ]:


len(df)


# **TASK: Create a Series that displays the total count of missing values per column.**

# In[ ]:


# CODE HERE


# In[ ]:


df.isnull().sum()


# **TASK: Convert this Series to be in term of percentage of the total DataFrame**

# In[ ]:


# CODE HERE


# In[ ]:


100* df.isnull().sum()/len(df)


# **TASK: Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook.**

# In[ ]:


# CODE HERE


# In[ ]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# **TASK: How many unique employment job titles are there?**

# In[ ]:


# CODE HERE


# In[ ]:


df['emp_title'].nunique()


# In[ ]:


df['emp_title'].value_counts()


# **TASK: Realistically there are too many unique job titles to try to convert this to a dummy variable feature. Let's remove that emp_title column.**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.drop('emp_title',axis=1)


# **TASK: Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.**

# In[ ]:


# CODE HERE


# In[ ]:


sorted(df['emp_length'].dropna().unique())


# In[ ]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[ ]:


plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)


# **TASK: Plot out the countplot with a hue separating Fully Paid vs Charged Off**

# In[ ]:


# CODE HERE


# In[ ]:





# In[ ]:





# In[ ]:





# **CHALLENGE TASK: This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan. There are a multitude of ways to create this Series. Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html). This may be tricky, refer to solutions if you get stuck on creating this Series.**

# In[ ]:


# CODE HERE


# In[ ]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[ ]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[ ]:


emp_len = emp_co/emp_fp


# In[ ]:


emp_len


# In[ ]:


emp_len.plot(kind='bar')


# **TASK: Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.drop('emp_length',axis=1)


# **TASK: Revisit the DataFrame to see what feature columns still have missing data.**

# In[ ]:





# In[ ]:


df.isnull().sum()


# **TASK: Review the title column vs the purpose column. Is this repeated information?**

# In[ ]:


# CODE HERE


# In[ ]:


df['purpose'].head(10)


# In[ ]:


df['title'].head(10)


# **TASK: The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.drop('title',axis=1)


# ---
# **NOTE: This is one of the hardest parts of the project! Refer to the solutions video if you need guidance, feel free to fill or drop the missing values of the mort_acc however you see fit! Here we're going with a very specific approach.**
# 
# 
# ---
# **TASK: Find out what the mort_acc feature represents**

# In[ ]:


# CODE HERE


# In[ ]:


feat_info('mort_acc')


# **TASK: Create a value_counts of the mort_acc column.**

# In[ ]:


# CODE HERE


# In[ ]:


df['mort_acc'].value_counts()


# **TASK: There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in, such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc**

# In[ ]:





# In[ ]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# **TASK: Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:**

# In[ ]:





# In[ ]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# **CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.**
# 
# [Helpful Link](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe) 

# In[ ]:


# CODE HERE


# In[ ]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[ ]:


total_acc_avg[2.0]


# In[ ]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[ ]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[ ]:


df.isnull().sum()


# **TASK: revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()


# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**
# 
# **TASK: List all the columns that are currently non-numeric. [Helpful Link](https://stackoverflow.com/questions/22470690/get-list-of-pandas-dataframe-columns-based-on-data-type)**
# 
# [Another very useful method call](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html)

# In[ ]:


# CODE HERE


# In[ ]:


df.select_dtypes(['object']).columns


# ---
# **Let's now go through all the string features to see what we should do with them.**
# 
# ---
# 
# 
# ### term feature
# 
# **TASK: Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().**

# In[ ]:


# CODE HERE


# In[ ]:


df['term'].value_counts()


# In[ ]:


# Or just use .map()
df['term'] = df['term'].apply(lambda term: int(term[:3]))


# ### grade feature
# 
# **TASK: We already know grade is part of sub_grade, so just drop the grade feature.**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.drop('grade',axis=1)


# **TASK: Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.**

# In[ ]:


# CODE HERE


# In[ ]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[ ]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[ ]:


df.columns


# In[ ]:


df.select_dtypes(['object']).columns


# ### verification_status, application_type,initial_list_status,purpose 
# **TASK: Convert these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[ ]:


# CODE HERE


# In[ ]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[ ]:





# ### home_ownership
# **TASK:Review the value_counts for the home_ownership column.**

# In[ ]:


#CODE HERE


# In[ ]:


df['home_ownership'].value_counts()


# **TASK: Convert these to dummy variables, but [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.**

# In[ ]:


#CODE HERE


# In[ ]:


df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# ### address
# **TASK: Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column.**

# In[ ]:


#CODE HERE


# In[ ]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# **TASK: Now make this zip_code column into dummy variables using pandas. Concatenate the result and drop the original zip_code column along with dropping the address column.**

# In[ ]:





# In[ ]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# ### issue_d 
# 
# **TASK: This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.**

# In[ ]:


#CODE HERE


# In[ ]:


df = df.drop('issue_d',axis=1)


# ### earliest_cr_line
# **TASK: This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.**

# In[ ]:


#CODE HERE


# In[ ]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[ ]:


df.select_dtypes(['object']).columns


# ## Train Test Split

# **TASK: Import train_test_split from sklearn.**

# In[ ]:


from sklearn.model_selection import train_test_split


# **TASK: drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

# In[ ]:


# CODE HERE


# In[ ]:


df = df.drop('loan_status',axis=1)


# **TASK: Set X and y variables to the .values of the features and label.**

# In[ ]:


#CODE HERE


# In[ ]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# ----
# ----
# 
# # OPTIONAL
# 
# ## Grabbing a Sample for Training Time
# 
# ### OPTIONAL: Use .sample() to grab a sample of the 490k+ entries to save time on training. Highly recommended for lower RAM computers or if you are not using GPU.
# 
# ----
# ----

# In[ ]:


# df = df.sample(frac=0.1,random_state=101)
print(len(df))


# **TASK: Perform a train/test split with test_size=0.2 and a random_state of 101.**

# In[ ]:


#CODE HERE


# In[ ]:


X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Normalizing the Data
# 
# **TASK: Use a MinMaxScaler to normalize the feature data X_train and X_test. Recall we don't want data leakge from the test set so we only fit on the X_train data.**

# In[ ]:


# CODE HERE


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train_unscaled)


# In[ ]:


X_test = scaler.transform(X_test_unscaled)


# # Creating the Model
# 
# **TASK: Run the cell below to import the necessary Keras functions.**

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# **TASK: Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron. OPTIONAL: Explore adding [Dropout layers](https://keras.io/layers/core/) [1](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) [2](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab)**

# In[ ]:


# CODE HERE

# Choose whatever number of layers/neurons you want.

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Remember to compile()


# In[ ]:


model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# model.add(Dense(512,  activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(256,  activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(128,  activation='relu'))
# model.add(Dropout(0.2))

# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy'])

# class_weight= {0:1, 1:1}
# with_weigths=True
# if (with_weigths):
#     neg, pos = np.bincount(y_train)
#     total = neg + pos
#     print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
#     total, pos, 100 * pos / total))
#     weight_for_0 = (1 / neg)*(total)/2.0 
#     weight_for_1 = (1 / pos)*(total)/2.0
#     class_weight = {0: weight_for_0, 1: weight_for_1}
#     print('Weight for class 0: {:.2f}'.format(weight_for_0))
#     print('Weight for class 1: {:.2f}'.format(weight_for_1))
        
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x=X_train, 
          y=y_train, 
          epochs=2,
#           class_weight=class_weight,
          batch_size=256,
          validation_data=(X_test, y_test), 
          callbacks = [early_stop]
          )

import kr_helper_funcs as kr
kr.show_plots(model.history.history)


# **TASK: Fit the model to the training data for at least 25 epochs. Also add in the validation data for later plotting. Optional: add in a batch_size of 256.**

# In[ ]:


# exp = explainer.explain_instance(X_test[4], model.predict, num_features=20, top_labels=1)
# exp.show_in_notebook(show_table=True, show_all=False)


# In[ ]:


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:80% !important; }</style>"))

import shap
import lime
from lime import lime_tabular
explainer = lime_tabular.LimeTabularExplainer(X_train)
exp = explainer.explain_instance(X_test[4], model.predict, num_features=20, top_labels=1)
exp.show_in_notebook(show_table=True, show_all=False)


# **TASK: OPTIONAL: Save your model.**

# In[ ]:


# def flatten_predict(i):
#     predictions = model.predict_proba(i)
#     x = np.zeros((predictions.shape[0], 1))
#     probability = (x+1) - predictions
#     return np.append(predictions, probability, axis = 1)

# def predict_fn(x):
#     preds = model.predict(x).reshape(-1, 1)
#     p0= 1- preds
#     return np.hstack((p0, preds))

# def explain_row(row, actual_prediction):
# #     print (row)
#     scaled_row = scaler.transform(row.values.reshape(1,78))
# #     print(scaled_row)

    
#     prediction = (model.predict_classes(scaled_row))[0][0]
#     probability =(model.predict_proba(scaled_row))
    
#     print ("Probability: {} Predict_fn: {} flatten_predict: {} Predicted class:{} Actual class:{}"
#            .format(probability, predict_fn(scaled_row), flatten_predict(scaled_row), prediction, actual_prediction))
    
#     explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=['paid', 'unpaid'])       
#     exp = explainer.explain_instance(scaled_row[0], predict_fn, num_features=10, 
#                                       top_labels=10) #labels=list(df.columns)
#     exp.show_in_notebook(show_table=True, show_all=False)
#     return exp.as_map()

# def recreate_row(row, map_values, no_exp):
#     scaled_row = scaler.transform(row.values.reshape(1,78))
#     predicted_class = (model.predict_classes(scaled_row))[0][0]
#     tuple_list = map_values[predicted_class]
#     important_columns = [a for (a,b) in tuple_list if b>0][:no_exp]
# #     if (predicted_class == 0):
# #         important_columns = [a for (a,b) in tuple_list if b<=0][:no_exp]
# #         print(important_columns)
# #     else:
# #         important_columns = [a for (a,b) in tuple_list if b>0][:no_exp]
# #         print(important_columns)
    
#     for c in important_columns:
#         row[c] = df[df['loan_repaid'] != predicted_class].mean()[c]
# #         row[c] = df.mean()[c]
# #     print(row)
#     return row
        

    
# row_number = 24

# row = df.drop('loan_repaid', axis = 1).iloc[row_number]
# map_values = explain_row(row, df.iloc[row_number]["loan_repaid"])
# print(map_values)
# new_row = recreate_row(row, map_values, 3)
# map_values = explain_row(new_row, df.iloc[row_number]["loan_repaid"])


# In[ ]:


def predict_fn(x):
    preds = model.predict(x).reshape(-1, 1)
    p0= 1- preds
    return np.hstack((p0, preds))

def explain_row_lime(scaled_row, explainer, nsamples = 100, verbose = 0):
    if nsamples == "auto":
        exp = explainer.explain_instance(scaled_row, predict_fn, num_features=10, top_labels=30) #labels=list(df.columns)
    else:
        exp = explainer.explain_instance(scaled_row, predict_fn, num_features=10, top_labels=30, num_samples=nsamples) #labels=list(df.columns)

    if (verbose == 1):
        exp.show_in_notebook(show_table=True, show_all=False)
        
    return exp.as_map()

def explain_row_shap(scaled_row, explainer, nsamples = 100, verbose = 0):
    shap_values = explainer.shap_values(scaled_row.reshape(1,78), nsamples=nsamples, l1_reg = "num_features(30)")
    
    if (verbose == 1):
        shap.decision_plot(explainer.expected_value[0], shap_values[0][0,:], scaled_row ,feature_names = list(df.drop('loan_repaid',axis=1).columns), link="logit")
    
    map_values ={}
    for class_value in range(len(shap_values)):
        s = shap_values[class_value][0]
        sorted_indices=sorted(range(len(s)), key=lambda k: s[k], reverse=True)
#         print(sorted_indices)
        ordered_list = [(a, shap_values[class_value][0][a]) for a in sorted_indices if shap_values[class_value][0][a]>0]
        map_values[class_value] = ordered_list
#     print(map_values)
    return map_values

def explain_row_random(scaled_row, explainer, nsamples=100, verbose=0):
    random_list = {}
    random_list[0]=[(i, random.uniform(0.0, 1.0)) for i in range(78)]
    random_list[1]=[(i, random.uniform(0.0, 1.0)) for i in range(78)]

    map_values = {}
    for class_value in range(len(random_list)):
        s = random_list[class_value]
        sorted_indices = sorted(range(len(s)), key=lambda k: s[k][1], reverse =True)
        #         print(sorted_indices)
        ordered_list = [(a, random_list[class_value][a][1]) for a in sorted_indices if random_list[class_value][a][1] > 0]
        map_values[class_value] = ordered_list
    #     print(map_values)
    return map_values



def recreate_row(neutral_points, unscaled_row, map_values, predicted_class, no_exp=3, verbose = 0):
    tuple_list = map_values[predicted_class]
    important_columns = [a for (a,b) in tuple_list if b>0][:no_exp]

    for c in important_columns:
        unscaled_row[c] = neutral_points[c]
        
    if (verbose == 1):
        print(pd.DataFrame(unscaled_row))
        
    return scaler.transform(pd.DataFrame(unscaled_row).values.reshape(1,78))

def calculate_probability_diff(row_number,neutral_points, explainer, no_exp=3, verbose = 0, which_explainer = 'lime', nsamples=100):
    scaled_row = X_test[row_number]
    unscaled_row = X_test_unscaled[row_number]

    reshaped_scaled_row = pd.DataFrame(scaled_row).values.reshape(1,78)
    original_class = model.predict_classes(reshaped_scaled_row)[0][0]
    original_probability =model.predict_proba(reshaped_scaled_row)
    original_predict_fn = predict_fn(reshaped_scaled_row)

    if(original_class != y_test[row_number]):
        return

    if (which_explainer == 'lime'):
        map_values = explain_row_lime(scaled_row, explainer, nsamples = nsamples, verbose = verbose)
    elif (which_explainer == 'shap'):
        map_values = explain_row_shap(scaled_row, explainer, nsamples = nsamples, verbose = verbose)
    elif (which_explainer == 'random'):
        map_values = explain_row_random(scaled_row, explainer, nsamples = nsamples, verbose = verbose)

    if (verbose == 1):
        print(map_values)

    new_row = recreate_row(neutral_points, unscaled_row, map_values, original_class, no_exp, verbose)
    new_class = model.predict_classes(new_row)[0][0]
    new_probability = model.predict_proba(new_row)
    new_predict_fn = predict_fn(new_row)
    
    if (verbose == 1):
        print ("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
               .format(original_probability, original_predict_fn, original_class, y_test[row_number]))

        print ("Probability: {} Predict_fn: {} Predicted class:{} Actual class:{}"
               .format(new_probability, new_predict_fn, new_class, y_test[row_number]))

#     map_values = explain_row(new_row[0], verbose)

    return ((original_predict_fn - new_predict_fn)[0][original_class], original_class, original_class!=new_class)


def calculate_values(number_of_rows = 100, number_of_exaplanations = 11, which_explainer = 'random', nsampleslist = [100], verbose = 0):
    if which_explainer == 'lime':
        explainer = lime_tabular.LimeTabularExplainer(X_train, training_labels=['paid', 'unpaid'])
    elif which_explainer == 'shap':
        explainer = shap.KernelExplainer(predict_fn, X_train[0:1000])
    elif which_explainer == 'random':
        explainer = None

    neutral_points = ((df[df['loan_repaid'] != 0].mean() + df[df['loan_repaid'] != 1].mean())/2).drop('loan_repaid')
    results = []

    predicted_classes = model.predict_classes(X_test[:number_of_rows])

    for nsamples in nsampleslist:
        for no_exp in range (number_of_exaplanations):
            print ("Explanation number:---------------------------{}---------------------------".format(no_exp))
            for row_number in range(number_of_rows):
                print ("Row Number {}".format(row_number))
                if (predicted_classes[row_number][0] != y_test[row_number]):
                    print ("Predicted and actual classes are different, skip")
                    continue;

                start_time = time.time()
                confidence_diff, original_class, class_change = calculate_probability_diff(row_number,neutral_points, explainer = explainer, no_exp= no_exp, verbose = verbose, which_explainer = which_explainer, nsamples=nsamples)
                total_time = time.time() - start_time
                results.append((confidence_diff, original_class, class_change, no_exp, nsamples, which_explainer, total_time))
                with open('lending-club-values.csv', 'a',  newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([confidence_diff, original_class, class_change, no_exp ,nsamples, which_explainer,total_time])
                    # fd.write(
                    #     confidence_diff + ',' + class_change + ',' + no_exp + ',' + nsamples + ',' + which_explainer + ',' + total_time)

    print (results)
    return results


import csv
import os

#add headers
if os.path.exists("lending-club-values.csv"):
    os.remove('lending-club-values.csv')
with open('lending-club-values.csv', 'a',  newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(["confidence_diff", "original_class", "class_change", "no_features", "nsamples", "explainer", "time"])

res_shap = calculate_values(number_of_rows = 1, number_of_exaplanations = 2, which_explainer = 'shap', nsampleslist = [100, 1000, "auto"]);
res_lime = calculate_values(number_of_rows = 1, number_of_exaplanations = 2, which_explainer = 'lime', nsampleslist = [100, 1000, "auto"]);
res_random = calculate_values(number_of_rows = 1, number_of_exaplanations = 2, which_explainer = 'random', nsampleslist = [100, 1000, "auto"]);

res_shap = calculate_values(number_of_rows = 200, number_of_exaplanations = 11, which_explainer = 'shap', nsampleslist = [100, 1000, "auto"]);
res_lime = calculate_values(number_of_rows = 200, number_of_exaplanations = 11, which_explainer = 'lime', nsampleslist = [100, 1000, "auto"]);
res_random = calculate_values(number_of_rows = 200, number_of_exaplanations = 11, which_explainer = 'random', nsampleslist = [100, 1000, "auto"]);

import pandas as pd
pd.DataFrame.from_dict(res_shap).to_csv('shap_values.csv',index=False)
pd.DataFrame.from_dict(res_shap).to_csv('lime_values.csv',index=False)

# compare lime with random feature replacement
# instead of mean of two sides, use an standard deviation
# create plots specific for each individual class
# no of samples created for explanations
# only use points correctly predicted by the model
# plot values that are negative - where the change increased the prediction instead of decreasing it

# In[ ]:


res = pd.DataFrame(results, columns =["confidence_diff", "class_change", "no_features"])
res.head(50)


# In[ ]:


sns.countplot(x="class_change",data=res)


# In[ ]:


avg = res.groupby("no_features")["confidence_diff"].mean()
plt.figure(figsize=(10,4))
# plt.set_title('Training & Validation Loss')
no_of_features = list(range(0, len(avg) ))
plt.plot( no_of_features, avg, color='navy', marker='o', linestyle='-', label='Confidence level change vs no of features')
plt.legend(loc='best')


# In[ ]:


# x, y, hue = "confidence_diff", "class_change", "no_features"
# prop_df = (res[x]
#            .groupby(res[hue])
#            .value_counts(normalize=True)
#            .rename(y)
#            .reset_index())
# plt.figure(figsize=(10,4))
# sns.barplot(x=x, y=y, hue=hue, data=prop_df)#, ax=axes[1])


# In[ ]:


plt.figure(figsize=(15,4))
sns.countplot(x='no_features',data=res,order=no_of_features,hue='class_change')


# In[ ]:


avg = res[res['confidence_diff'] == 0.0].groupby("no_features")["confidence_diff"].count() / res.groupby("no_features")["confidence_diff"].count()
plt.figure(figsize=(10,4))
# plt.set_title('Training & Validation Loss')
no_of_features = list(range(0, len(avg) ))
plt.plot( no_of_features, avg, color='navy', marker='o', linestyle='-', label='Confidence level change vs no of features')
plt.legend(loc='best')


# In[ ]:


import shap
# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
shap.initjs()


# In[ ]:





# In[ ]:





# In[ ]:


from tensorflow.keras.models import load_model
model.save('lending-club.h5')  


# In[ ]:


c


# # Section 3: Evaluating Model Performance.
# 
# **TASK: Plot out the validation loss versus the training loss.**

# In[ ]:


# plot the SHAP values for the Setosa output of the first instance
shap.decision_plot(explainer.expected_value[0], shap_values[0][3, :])


# In[ ]:





# In[ ]:


explainer = shap.KernelExplainer(predict_fn, X_train[0:1000])
shap_values = explainer.shap_values(X_test[0].reshape(1,78), nsamples=100)


# In[ ]:


shap.force_plot(explainer.expected_value[0], shap_values[0][3,:], df.drop('loan_repaid',axis=1).columns, link="logit")


# In[ ]:


nsamples = 100
row_num = 10
shap_values = explainer.shap_values(X_test[row_num].reshape(1,78), nsamples=nsamples, l1_reg = "num_features(10)",)
print(shap_values)
shap.decision_plot(explainer.expected_value[0], shap_values[0][0,:], X_test[row_num,:],feature_names = list(df.drop('loan_repaid',axis=1).columns), link="logit")


# **TASK: Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[ ]:



np.argmax(shap_values[0])


# In[ ]:


print(shap_values[1][0])
s = shap_values[1][0]
print(shap_values[1][0][28])


# In[ ]:




explain_row_shap(X[10], 1)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


len(shap_values)


# In[ ]:


predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))


# In[ ]:


confusion_matrix(y_test,predictions)


# **TASK: Given the customer below, would you offer this person a loan?**

# In[ ]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[ ]:


# CODE HERE


# In[ ]:


model.predict_classes(new_customer.values.reshape(1,78))


# **TASK: Now check, did this person actually end up paying back their loan?**

# In[ ]:


# CODE HERE


# In[ ]:


df.iloc[random_ind]['loan_repaid']


# # GREAT JOB!
