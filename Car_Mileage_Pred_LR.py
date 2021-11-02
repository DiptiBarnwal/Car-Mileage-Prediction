#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np   
from sklearn.linear_model import LinearRegression
import pandas as pd    
import matplotlib.pyplot as plt   
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


# reading the CSV file into pandas dataframe
df = pd.read_csv("/content/drive/My Drive/Regression Models_Mahesh Anand/car_data.csv")  


# In[6]:


df.head()


# In[7]:


df.columns


# In[36]:


df['gear'].value_counts()


# In[37]:


plt.scatter(df['gear'],df['mpg'])


# In[8]:


df=df.drop('Unnamed: 0',axis=1)
df.head()


# In[ ]:


df.head()


# In[ ]:


sns.pairplot(df,diag_kind='kde')


# In[9]:


#Create dummy columns to categorical columns
B1=pd.get_dummies(df['cyl'])
B2=pd.get_dummies(df['gear'])
B3=pd.get_dummies(df['carb'])
B=pd.concat([df,B1,B2,B3],axis=1)
B.head()


# In[10]:


B_update=B.drop(['cyl','gear','carb'],axis=1)
B_update.head()


# In[12]:


df['gear'].value_counts()


# In[ ]:


B_update.columns


# In[13]:


B_update.columns=['mpg','disp','hp','drat','wt','qsec','vs','am','c4','c6','c8','g3','g4','g5','cb1','cb2','cb3','cb4','cb6','cb8']
B_update.head()


# In[14]:


X=B_update.drop('mpg',axis=1)
Y=B_update['mpg']
X.head()


# In[ ]:


cols = list(X.columns)
X = X[cols]
X.head()


# In[15]:


xc =sm.add_constant(X)


# In[17]:


model = sm.OLS(Y,xc).fit()
model.summary()


# In[40]:


#cols = list(X.columns)
#X = X[cols]
xc = sm.add_constant(X_final)
model = sm.OLS(Y,xc).fit()
model.summary()


# In[30]:


p = pd.Series(model.pvalues.values[1:],index = cols)   
max(p)


# In[31]:


p.idxmax()


# In[ ]:


p = pd.Series(model.pvalues.values[1:],index = cols)  
p


# In[26]:


feature_with_p_max = p.idxmax()
feature_with_p_max


# In[27]:


cols.remove(feature_with_p_max)


# In[ ]:


p.idxmax()


# In[28]:


cols


# In[32]:


cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X = X[cols]
    xc = sm.add_constant(X)
    model = sm.OLS(Y,xc).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


df.columns


# In[38]:


X_final=B_update[selected_features_BE]
X_final.head()


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


xtrain,xtest,ytrain,ytest=train_test_split(X_final,Y,test_size=0.30,random_state=0)


# In[ ]:


xtrain.shape,ytrain.shape,xtest.shape,ytest.shape


# In[51]:


model=LinearRegression()
model.fit(xtrain,ytrain)
y_pred_test=model.predict(xtest)
y_pred_test


# In[52]:


residue_test=ytest-y_pred_test


# In[53]:


y_pred_train=model.predict(xtrain)  #to check the assumptions manually instead of OLS


# In[54]:


#training records residue
residue_train=ytrain-y_pred_train


# In[55]:


residue_train.skew()


# In[57]:


from sklearn import metrics


# In[56]:


#RMSE score of LR model for the test records
rmse_test=np.sqrt(np.mean((ytest-y_pred_test)**2))
print(rmse_test)


# In[60]:


#RMSE score of LR model for the training records
rmse_train=np.sqrt(np.mean((ytrain-y_pred_train)**2))
print(rmse_train)


# In[59]:


mse=metrics.mean_squared_error(ytest,y_pred_test)
rmse=np.sqrt(mse)
print(rmse)


# For Linear Regression, we need to check if the 5 major assumptions hold.
# 
# 1. No Auto correlation among the residues
# 2. Linearity of variables
# 3. Normality of error terms
# 4. No Heteroscedacity
# 5. No strong MultiCollinearity

# 1) No Auto correlation. 
# 
# Test needed : Durbin- Watson Test.
# 
# - It's value ranges from 0-4. If the value of Durbin- Watson is Between 0-2, it's known as Positive Autocorrelation.
# - If the value ranges from 2-4, it is known as Negative autocorrelation.
# - If the value is exactly 2, it means No Autocorrelation.
# - For a good linear model, it should have low or no autocorrelation.
# 

# In[42]:


xc =sm.add_constant(X_final)
final_model = sm.OLS(Y,xc).fit()


# In[ ]:


final_model.summary()


# In[43]:


# Check the Asumptions of Linear Regression
import statsmodels.tsa.api as smt

acf = smt.graphics.plot_acf(final_model.resid, lags=10 , alpha=0.05)
acf.show()


# 2) The second assumption is the Normality of Residuals / Error terms.
# 
# For this we prefer the Jarque Bera test. For a good model, the residuals should be normally distributed.
# The higher the value of Jarque Bera test , the lesser the residuals are normally distributed.
# We generally prefer a lower value of jarque bera test.
# 
# The Jarque–Bera test is a goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution. A large value for the jarque-bera test indicates non normality.
#     
# The jarque bera test tests whether the sample data has the skewness and kurtosis matching a normal distribution.
# 

# In[44]:


#sample size is >2000 & <5000
#If sample size >5000 (Anderson_Darling Test)
from scipy import stats
print(stats.jarque_bera(final_model.resid))


# In[45]:


#sample size <2000
from scipy import stats
print(stats.shapiro(final_model.resid))


# In[46]:


import seaborn as sns

sns.distplot(final_model.resid)


# In[ ]:


y_pred_train=model.predict(xtrain)
y_pred_train
res=(ytrain-y_pred_train)
res.skew()


# ##### Asssumption 3 - Linearity of residuals
# We can plot the observed values Vs predicted values and see the linearity of residuals.
# 

# In[ ]:


plt.plot(ytrain,y_pred_train,'*')


# ##### Assumption 4 -  Homoscedasticity
# Homoscedacity :: If the variance of the residuals are symmetrically distributed across the regression line , then the data is said to homoscedastic.
# 
# Heteroscedacity :: If the variance is unequal for the residuals across the regression line, then the data is said to be heteroscedastic. 
# This can be visually noticed in sns scatter plot

# ##### Assumption 5- NO  MULTI COLLINEARITY
# Multicollinearity effect can be be observed from correlation matrix, however the treatment for multicollinearity among independent variables can be effectively done through PCA technique (which we will be learning in future course)
