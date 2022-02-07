# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 15:20:38 2022

@author: Dell
"""


#%%
import pandas as pd
df=pd.read_csv('car data.csv')
print(df.head)
#%%
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
#%%
#checking missing value
print(df.isnull().sum())
#%%
print(df.describe())
#%%
final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
print(final_dataset.head())
#%%
#Adding one coloumn "no years" representing Year and Current year columns
final_dataset['Current_Year'] = 2022
final_dataset['no_years'] = final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.drop(['Year'], axis = 1, inplace = True)
final_dataset.drop(['Current_Year'], axis = 1, inplace = True)
#This inplace = True because the operation happens in place or there will be a problem 
#if you don't understant watch this https://www.youtube.com/watch?v=lNMJ6l0XbPA&t=201s
print(final_dataset.head())
#%% I want to convert seller type, fuel type, transmission to some other values
# we are gonna do it by get dummy method inside pandas to convert them to hot encoded
# we choose them because they have less features categories
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
#drop_first:To Remove first level to get k-1 dummies out of k categorical levels.
print (final_dataset.head())
#%% find correlation
corrmat = final_dataset.corr()
print(corrmat)
#%% visualize it in diagrams
import seaborn as sns
sns.pairplot(final_dataset)
#%%
#because the last diagram didn't show enough info
#we gonna represent it(corr) in a form of a heat map
import matplotlib.pyplot as plt
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#%% 
#independant and dependent features
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
print(X.head())
print(y.head())
#%%
#features importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
#%%
print(model.feature_importances_)
#%%
#plot graph of feature importances in a better visualization way
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest().plot(kind='barh')
plt.show()
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 )
print(X_train.shape)
#%%
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
#%%
#Hyperparameters
import numpy as np
from sklearn.model_selection import RandomizedSearchCV #to select the best parameters value
#Randomized Search CV
#Hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
#%%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
#%%
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
#here we go 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
#%%
rf_random.fit(X_train,y_train)
#%%
predictions=rf_random.predict(X_test)
print(predictions)
#%%
#compare our prediction to the true labels
sns.distplot(y_test-predictions)
#%%
plt.scatter(y_test,predictions)

#%%











