# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:42:34 2018

@author: Varun
"""

"""
Created on Sat Nov 24 11:11:00 2018

@author: Varun
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error as abse
from sklearn.metrics import f1_score as f1
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import seaborn as sns

plt.style.use("ggplot")
folder = "E:/Acads/9th sem/Citi hiring chal/"
df = pd.read_csv(folder+"train.csv")
dft = pd.read_csv(folder+"test.csv")

# Identifying columns in test data, which are present in the training data
for i in df.columns : 
  if i not in dft.columns : print(i)
for i in dft.columns : 
  if i not in df.columns : print(i)

# Identified that 'INSPECTION NO.' is not present in the test data
df.drop('INSPECTION NO.', axis =1, inplace = True)

# Identifying the columns with least number of unique categorical values
unic = pd.Series([len(df.iloc[:,i].unique()) for i in range(len(df.columns))], index = df.columns)
unic.sort_values(inplace = True)
print(unic)

# Separating the Target variable and remove unnecesasary columns from data
dfy = df["CRITICAL FLAG"]
df = df.drop(["CRITICAL FLAG", "RECORD DATE", "UIDX"], axis =1)

# Decompose violation code into violation number and character into two features
df['VIOLATION CODE'] = df['VIOLATION CODE'].fillna("99Z")
df["V NUM"] = [int(x[:2]) for x in df['VIOLATION CODE']] #new feature
df["V CHR"] = [x[-1] for x in df['VIOLATION CODE']]  #new feature

#Identifying the columns with least number of unique categorical values 
unic = pd.Series([len(df.iloc[:,i].unique()) for i in range(len(df.columns))], index = df.columns)
unic.sort_values(inplace = True)

# Find the Nan count in each column
nanlist = df.isnull().sum()

# Helper function to plot Cross Tables in percentages
def plotcross(dfy, dfx):
  cross = pd.crosstab(dfy, dfx, dropna = False)
  cross = cross/cross.sum()
  sns.heatmap(cross, cmap = "Blues")

# Helper function for One Hot Encoding a column
def oneHotEnc(df, col):
  dfdum = pd.get_dummies(df[col],prefix =col,  dummy_na = True)
  df = pd.concat([df.drop([col], axis =1), dfdum], axis = 1)
  return df

# Start checking the tables and iteratively go through one hot encoding as required
cross = pd.crosstab(dfy, df['ACTION'], dropna = False)
plotcross(dfy, df['ACTION'])
df = oneHotEnc(df, 'ACTION')

plotcross(dfy, df['BORO'])
df['BORO']= df['BORO'].apply(lambda x : 1 if x=='Missing' else 0)


cross = pd.crosstab(dfy, df[unic.index[2]], dropna = False)
plotcross(dfy, df[unic.index[2]])
df = oneHotEnc(df, unic.index[2])

df[unic.index[3]] = df[unic.index[3]].apply(lambda x : x.strip())
cross = pd.crosstab(dfy, df[unic.index[3]], dropna = False)
plotcross(dfy, df[unic.index[3]])
df = oneHotEnc(df, unic.index[3])

cross = pd.crosstab(dfy, df[unic.index[4]], dropna = False)
plotcross(dfy, df[unic.index[4]])

# From the cross table plot (also observing the cross table) we arrive that many 
# of these variables can be clustered. Hence we cluster them taking inferernces from
# the plot
df["V NUM"] = [0 if x<=7 else x for x in df['V NUM']]
cross = pd.crosstab(dfy, df[unic.index[5]], dropna = False)
plotcross(dfy, df[unic.index[4]])

df["V NUM"] = [1 if x==15 else x for x in df['V NUM']]
df["V NUM"] = [2 if x==22 else x for x in df['V NUM']]
df["V NUM"] = [3 if x==99 else x for x in df['V NUM']]
df["V NUM"] = [4 if x>3 else x for x in df['V NUM']]

cross = pd.crosstab(dfy, df[unic.index[4]], dropna = False)
plotcross(dfy, df[unic.index[4]])
df = oneHotEnc(df, unic.index[4])

# Let us now take all the columns we encoded and try to fit a model
dfX = df.drop(unic.index[5:], axis =1)

le = LabelEncoder()
dfY = le.fit_transform(dfy)

Xtr, Xts, ytr, yts = train_test_split(dfX, dfY, test_size = 0.3) # 30% testing split
reg = XGBClassifier(n_jobs = 6)
reg.fit(Xtr, ytr)
print(reg.score(Xts, yts))
# We obtain a very high cross validation accuracy of 99.97%. We hence stop adding
# more variables and appropriately transform the test data and make predictions

ind = dft["UIDX"]
dft = dft.drop([ "RECORD DATE", "UIDX"], axis =1)

dft['VIOLATION CODE'] = dft['VIOLATION CODE'].fillna("99Z")
dft["V NUM"] = [int(x[:2]) for x in dft['VIOLATION CODE']]
dft["V CHR"] = [x[-1] for x in dft['VIOLATION CODE']]

dft = oneHotEnc(dft, 'ACTION')
dft['BORO']= dft['BORO'].apply(lambda x : 1 if x=='Missing' else 0)
dft = oneHotEnc(dft, unic.index[2])

# There is some ambiguity with 'Inspection Type' in testing data whih we correct
dft[unic.index[3]] = dft[unic.index[3]].apply(lambda x : x.split("/")[0].strip()  if type(x)!= float else np.nan) 
dft[unic.index[3]].fillna("Cycle Inspection", inplace = True)
dft = oneHotEnc(dft, unic.index[3])

dft["V NUM"] = [0 if x<=7 else x for x in dft['V NUM']]
dft["V NUM"] = [1 if x==15 else x for x in dft['V NUM']]
dft["V NUM"] = [2 if x==22 else x for x in dft['V NUM']]
dft["V NUM"] = [3 if x==99 else x for x in dft['V NUM']]
dft["V NUM"] = [4 if x>3 else x for x in dft['V NUM']]

dft = oneHotEnc(dft, unic.index[4])

dftX = dft.drop(unic.index[5:], axis =1)

# Final predictions after transformation and we output the csv file submitted
ypred = reg.predict(dftX)
output = pd.DataFrame( le.inverse_transform(ypred), index = ind.values, columns = ["CRITICAL FLAG"])
output["UIDX"] = output.index
output = output[["UIDX", "CRITICAL FLAG"]]
output.to_csv(folder+"output.csv", index = False )






