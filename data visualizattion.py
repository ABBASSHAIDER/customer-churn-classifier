#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df =pd.read_csv('c:/csv/Churn/Churn_Modelling.csv')

df.head()


# In[3]:


df=df.iloc[:,3:]
df


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# # Using Heat map to check the null values in dataset

# In[9]:


sn.heatmap(df.isnull(),cmap="viridis")


# In[ ]:





# In[10]:


fig , ax =plt.subplots(2,2)
ax = ax.flatten()
fig.set_size_inches(14, 10)
sn.distplot(df.EstimatedSalary,color='#2D008E',ax=ax[0])
sn.distplot(df.CreditScore,color='#F9971A',ax=ax[1])
sn.distplot(df.Balance,color='#242852',ax=ax[2])
sn.distplot(df.Age,color='#242852',ax=ax[3])


# In[11]:


fig_1 , ax_1 =plt.subplots(2,2)
ax_1 = ax_1.flatten() 
fig_1.set_size_inches(16, 10)
sn.stripplot(y=df.EstimatedSalary,x=df.Exited,palette="YlOrBr",ax=ax_1[0])
sn.boxplot(y=df.EstimatedSalary,x=df.Exited,palette="YlOrBr",ax=ax_1[1]) 
sn.stripplot(y=df.EstimatedSalary,x=df.Tenure,palette="mako",ax=ax_1[2]) 
sn.boxplot(y=df.EstimatedSalary,x=df.Tenure,palette="mako",ax=ax_1[3])


# # Remove outlier by using standard deviation

# #upper limit standard deviation of CreditScore

# In[12]:


upper_limit_CreditScore= df["CreditScore"].mean() + 1.5*df["CreditScore"].std()
upper_limit_CreditScore


# #Lower limit standard deviation of CreditScore

# In[13]:


lower_limit_CreditScore = df["CreditScore"].mean() - 1.5*df["CreditScore"].std()
lower_limit_CreditScore


# #Upper limit standard deviation of EstimatedSalary

# In[14]:


upper_limit_EstimatedSalary= df["EstimatedSalary"].mean() + 1*df["EstimatedSalary"].std()
upper_limit_EstimatedSalary


# #Lower limit standard deviation of EstimatedSalary

# In[15]:


lower_limit_EstimatedSalary= df["EstimatedSalary"].mean() - 1.5*df["EstimatedSalary"].std()
lower_limit_EstimatedSalary


# In[16]:


df1=df[(df["CreditScore"]<upper_limit_CreditScore)&(df["CreditScore"]>lower_limit_CreditScore)]
df2=df1[(df1["EstimatedSalary"]<upper_limit_EstimatedSalary)&(df1["EstimatedSalary"]>lower_limit_EstimatedSalary)]
df1.shape,df2.shape


# In[17]:


fig_2 , (ax_21,ax_22) =plt.subplots(1,2)

fig_2.set_size_inches(12, 4)
ax_21.hist(df2.EstimatedSalary,color='#2D008E')
ax_22.hist(df2.CreditScore,color='#F9971A')


# In[18]:


fig_3 , ax_3 =plt.subplots(2,2)
ax_3 = ax_3.flatten()
fig_3.set_size_inches(16, 8)
sn.violinplot(y=df2.CreditScore,x=df2.Tenure,palette="mako",ax=ax_3[0])
sn.barplot(y="CreditScore",x="Tenure",data=df2,palette="mako",ci="sd",ax=ax_3[1])
sn.violinplot(y=df2.CreditScore,x=df2.NumOfProducts,palette="rocket",ax=ax_3[2]) 
sn.barplot(y="CreditScore",x="NumOfProducts",data=df2,palette="rocket",ci="sd",ax=ax_3[3])


# In[19]:


df2_churn_1=df2[df2["Exited"]==1]
df2_churn_0=df2[df2["Exited"]==0]
df2_churn_1.shape,df2_churn_0.shape


# In[20]:


df2.dtypes


# # Labeling the gender column by 0 and 1

# In[21]:


df3 = df2.convert_dtypes()
df3["Gender"] = np.where(df3["Gender"]=='Female',0,1)


# In[22]:


df3["Gender"]


# # creating a x dataset and y labels to train a model

# In[23]:


x=df3.drop(["Exited","Geography"],axis=1)
y=df3["Exited"]

y=y.astype('int')
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# # Standarized the train and test data to increase the accuracy

# In[24]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[25]:


x_train.shape,x_test.shape,y_train.shape


# # Apply Logistic Regression model

# In[26]:


from sklearn.linear_model import  LogisticRegression
lr = LogisticRegression()
model_logic = lr.fit(x_train,y_train)


# # Apply Random Forest Classifier

# In[27]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model_rfc = rfc.fit(x_train,y_train)


# # Apply Decision Tree Classifier

# In[28]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',max_depth=300)
model_dtc = dtc.fit(x_train,y_train)


# # Apply support vector Machine

# In[29]:


from sklearn.svm import SVC
svc = SVC()
model_svc = svc.fit(x_train,y_train)


# # Check the accuracy of all models by using Cross Validation Classifier

# In[30]:


from sklearn.model_selection import cross_val_score


# #logistic model accuracy

# In[31]:


scores_logic = cross_val_score(model_logic, x, y, cv=5)
scores_logic


# #random forest model accuracy

# In[32]:


scores_rfc = cross_val_score(model_rfc, x, y, cv=5)
scores_rfc


# #decision tree accuracy

# In[33]:


scores_dtc = cross_val_score(model_dtc, x, y, cv=5)
scores_dtc


# #SVC accuracy

# In[34]:


scores_svc = cross_val_score(model_svc, x, y, cv=5)
scores_svc


# #the accuracy results of show that random forest performs well. 

# # Using Confusion matrics to check the performance of random forest model

# In[35]:


y_predict = model_rfc.predict(x_test)


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm


# # Using heat map to plot confusion matrix. so that we can better understand the results

# In[37]:


plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predict')
plt.ylabel('Truth')


# #This heat map shows that the model predict  1159 times 0 and the actual value is also 0

# # Now i'm chect manually the result of my model 

# In[38]:


credit_score = 608
gender = 1 # which is male
age = 55
tenure = 6
balance = 83000
num_of_buy_product=12
has_a_creditcard = 1
is_active_member=0
salary=42000


arr=np.array([credit_score,gender,age,tenure,balance,num_of_buy_product,has_a_creditcard,is_active_member,salary])
arr_res=arr.reshape(1,-1)
a=model_rfc.predict(arr_res)
a


# #this result shows that the customer will exit
