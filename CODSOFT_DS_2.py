#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_csv("C:\\Users\\Smit\\OneDrive\\Desktop\\DATASET\\IMDb Movies India.csv")
df.head()


# In[3]:


print(df.shape)
print(df.info())
print(df.describe(include="all").T.head(15))


# In[4]:


target = "rating"
features = ["genre", "director", "actor_1", "actor_2", "actor_3", "budget", "duration"]

X = df[features]
y = df[target]


# In[5]:


numeric_features = ["budget", "duration"]
categorical_features = ["genre", "director", "actor_1", "actor_2", "actor_3"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# In[6]:


model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])


# In[7]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[8]:


feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["regressor"].feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:20]
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Top 20 Feature Importances")
plt.show()


# In[9]:


import joblib
joblib.dump(model, "movie_rating_model.joblib")


# In[ ]:




