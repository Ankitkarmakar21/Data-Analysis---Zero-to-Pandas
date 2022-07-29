#!/usr/bin/env python
# coding: utf-8

# # ANALYZING LOAN PREDICTION DATASET USING PYTHON
# 
# In this project,we are trying to analyze the data of a finance company who deals in different types of loans. They first check the eligibility with some checklist and the provide loans on the basis of it. The sheet contains data of approx 614 customers ranging from Urban to rural areas, graduate/Non graduate customers. 
# 
# **Here I have divided this entire project into three different section.**
# 1. Downloading the dataset and performing some cleaning(if required) and deriving some basic information of the data.
# 2. Visualizing the datasets using graphs 
# 3. Inferences and Future work that can be done using this dataset.
# 
# we have used kaggle as the source to download the data. We have used https://jovian.ai/learn/data-analysis-with-python-zero-to-pandas to learn about the basic libraries that we are going to use here.

# ## Downloading the Dataset
# 
# **here we will download our dataset.**
# **we are using kaggle as our source for our data.**

# In[1]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[2]:


# Change this
dataset_url = 'https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset' 


# In[3]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[4]:


# Change this
data_dir = './loan-prediction-problem-dataset'


# In[5]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[6]:


project_name = "analyzing loan prediction dataset using python" # change this (use lowercase letters and hyphens only)


# In[7]:


get_ipython().system('pip install jovian --upgrade -q')


# In[8]:


import jovian


# In[9]:


jovian.commit(project=project_name)


# ## READING DATASET
# #### We will import some important libraries which we will use in this project.

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df=pd.read_csv('./loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df


# In[12]:


df.describe()


# In[13]:


df.dtypes


# ## DATA CLEANING

# In[14]:


df.isna().sum()


# **We can see that Gender,Married,Self employed,Dependents have NA values. These columns are object type and we have replace these NA values by taking mode of the respective column.** 

# In[15]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)


# In[16]:


y=df['LoanAmount'].describe()
print("Mode of Loan Amount",df['LoanAmount'].mode())
print(y)


# **As mean is generally affected by outliers,we chose NA values of loan amount to be replaced by median of this column.**

# In[17]:


df['Loan_Amount_Term'].value_counts()


# **As Terms are mostly 360 months, we chose to replace NA values by mode of this column.**
# **Similary we chose to replace the credit history by mode of that column.**

# In[18]:


df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# **CHECK**

# In[19]:


df.isna().sum()


# ## EXPLORATORY ANALYSIS

# In[20]:


df["Loan_Status"].value_counts()


# In[21]:


df["Married"].value_counts()


# In[22]:


df['Gender'].value_counts()


# In[23]:


df['Education'].value_counts()


# In[24]:


df['Self_Employed'].value_counts()


# In[25]:


df['Property_Area'].value_counts()


# In[26]:


df['Credit_History'].value_counts()


# In[27]:


grp=df.groupby('Loan_Status')["Self_Employed"]


# In[28]:


grp.get_group("Y").value_counts()


# In[29]:


grp.get_group("N").value_counts()


# In[30]:


grp1=df.groupby('Loan_Status')["Education"]


# In[31]:


grp1.get_group("N").value_counts()


# In[32]:


grp2=df.groupby('Loan_Status')["ApplicantIncome"]


# In[33]:


grp2.get_group("N").value_counts().head(20)


# ## VISUALIZING DATA USING CHARTS
# #### We are using Matplotlib and seaborn libraries.

# In[34]:


import warnings
warnings.filterwarnings('ignore')

sns.set(font_scale = 1.1)
sns.set_style("darkgrid")
plt.figure(figsize=(18,12))
plt.subplot(3,3,1)
plt.subplots_adjust(wspace=1,hspace=.5)


a1=sns.countplot('Loan_Status',data=df,palette="nipy_spectral_r")
a1.bar_label(a1.containers[0])
a1.set_xlabel("Yes or No",size=15)
a1.set_ylabel("Count(Y/N)",size=15)


plt.subplot(3,3,2)
a2=sns.countplot('Gender',data=df,palette="plasma")
a2.bar_label(a2.containers[0])
a2.set_xlabel("Male or Female",size=15)
a2.set_ylabel("Count (M/F)",size=15);

plt.subplot(3,3,3)
a3=sns.countplot('Self_Employed',data=df,palette="summer_r")
a3.bar_label(a3.containers[0])
a3.set_xlabel("Whether self employed",size=15)
a3.set_ylabel("Count (Y/N)",size=15);

plt.subplot(3,3,4)
a4=sns.countplot('Property_Area',data=df,palette="cool")
a4.bar_label(a4.containers[0])
a4.set_xlabel("Property Area",size=15)
a4.set_ylabel("Count",size=15);


plt.subplot(3,3,5)
a5=sns.countplot('Credit_History',data=df,palette="YlGnBu_r")
a5.set_xlabel("Credit History",size=15)
a5.set_ylabel("Count",size=15);

plt.subplot(3,3,6)
a6=sns.countplot('Dependents',data=df,palette="cubehelix")
a6.set_xlabel("Dependents",size=15)
a6.set_ylabel("Count",size=15);

plt.show()



# ***Here we have plotted some graphs. We can see that Loan approved were more than loans rejected. The appliers are mostly Males and Working (Not self employed).Most of them have good credit history and the appliers are mostly from urban and semiurban areas so these loans can be loans for houses.***

# In[35]:


sns.set(font_scale = 1.1)
sns.set_style("darkgrid")
plt.figure(figsize=(8,4))
a7=sns.countplot('Loan_Amount_Term',data=df,palette="cividis_r")
a7.set_xlabel("Loan Amount Term",size=15)
a7.set_ylabel("Count",size=15);


# ***We can see that most loan amount term are for 360 months (i.e. 30 Years). We can reach to a conclusion that these are housing loans as we have seen above.***

# In[36]:


sns.set(font_scale = 1.1)
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
plot=sns.distplot(df['ApplicantIncome'],color='purple',kde=True);
sns.kdeplot(data=df, x="ApplicantIncome", color='black',ax=plot,lw='3',ls='--');


# ***Most of the appliers have income ranging from 0 to 20,000. Let's us see if there are any outliers.***

# In[37]:


sns.set(font_scale = 1.1)
sns.set_style("white")
plt.figure(figsize=(10,8))
sns.scatterplot(x=df['LoanAmount'],y=df['ApplicantIncome'],hue=df['Loan_Status'],sizes=(40, 400));


# ***Most of the Applicants have income between 0 to 20,000 with loan amount ranging from 0 to 300.***
# 
# ***We can see that a applier with income of 80000 had his loan rejected. This means that greater applicant income doesnot have affect on approval. We have to look on other criteria as to why the loan was rejected.***

# **Why was his loan rejected?
# We will answer this by filtering that specific row.**

# In[38]:


grp3=df.groupby('Loan_Status')['Credit_History','Dependents','Property_Area']


# In[39]:


grp3.get_group("Y").value_counts()


# In[40]:


x=df['ApplicantIncome']>80000


# In[41]:


df[x]


# ***We can find as why the loan was rejected for a applicant income with 81000***
# 
# 1.) Credit History is not good. Most of the approved applications had credit history of 1.0
# 
# 2.) Most of the loan which was approved had dependents of 0 and belonged to either urban or semi-urban area.

# **Lets Save our Work**

# In[42]:


import jovian


# In[43]:


jovian.commit()


# ## Asking and Answering Questions
# 
# We will try to answer some questions about the dataset to reach to a conclusion.
# 
# 

# #### Q1: What percentage of the total loan applied were approved?

#  <b style='color:Blue'> <font size="2.5">About 69% of the applied loans were approved. This means the more loans were approved than rejected.</b></font>

# #### Q2: Does loan approval depend on dependents of applicant?

#  <b style='color:Blue'> <font size="2.5"> The percentage of applicants with either 0 or 2 dependents have got their loan approved is higher.</b></font>

# #### Q3: What is the impact of credit history on loan approval?

#  <b style='color:Blue'> <font size="2.5">
#     Around 98% Loans that were approved had applicant with good credit history.
#    </b></font>
# 

# #### Q4: Does high applicant income result in loan approval?

#  <b style='color:Blue'> <font size="2.5">
#     No, Loan of Applicant with income of 81000 was rejected.
#    </b></font>
# 

# In[44]:


import jovian


# In[45]:


jovian.commit()


# # Inferences and Conclusion
# 
# 1.) About 69% of the applied loans were approved. This means the more loans were approved than rejected.
# 
# 2.) The loan approved for Non Self employed candidates are more than self employed candidates. but the loan rejected are also higher in non self employed candidates. Thus we cannot have any relation between loan approval with profession.
# 
# 3.) Married applicants are more than unmarried applicants (more than 65%).
# 
# 4.) Most of the appliers are from Urban and Semi-Urban areas. These are most likely loans applied for housing.
# 
# 5.) Most of the applied loans have terms more than 360 months. These loans indicate more likely loan against property and those with 180 months are more likely personal loans.
# 
# 6.) Most of the applicants have income from 0 to 20,000. Higher Loan Amount has no affect on loan approval.

# In[46]:


import jovian


# In[47]:


jovian.commit()


# ## References and Future Work
# 
# **Here our basic aim was to get some details about loan approved and loan rejected. The above dataset can be used to predict if a new person who applies for loan is eligible to get loan. This can be done using ML techniques.**
# 
# ### References
# 
# 1.) https://www.geeksforgeeks.org/seaborn-color-palette/ - **We used this to pick colours for our chart to make it attractive.**
# 
# 2.) https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/?ref=lbp - **Used this to have a overview of countplot.**
# 
# 3.) https://jovian.ai/learn/data-analysis-with-python-zero-to-pandas - **Use this to learn about pandas and other libraries in depth**

# In[48]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




