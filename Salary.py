#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error , r2_score


# In[2]:


df = pd.read_csv(r'Salary_Data.csv')
df 


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.dropna(inplace=True)


# In[7]:


df['Education Level'].value_counts()


# In[8]:


# Combining repeating values of education level

df['Education Level'].replace(["Bachelor's Degree","Master's Degree","phD"],["Bachelor's","Master's","PhD"],inplace=True)
df['Education Level'].value_counts()


# In[9]:


df['Job Title'].value_counts()


# In[10]:


# Reducing Job titles by omitting titles with less than 25 counts

job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count<=25]
job_title_edited.count()


# In[11]:


df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x )
df['Job Title'].nunique()


# In[12]:


df


# In[13]:


df['Job Title'].value_counts()


# In[22]:


# Outlier
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Data shape before salary outlier removal:", df.shape)
df = df[(df['Salary'] >= lower) & (df['Salary'] <= upper)]
print("Data shape after salary outlier removal:", df.shape)

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Salary'])
plt.title("Boxplot of Salary ")
plt.show()


# In[15]:


# Create a figure with three subplots
fig, ax = plt.subplots(3, 1, figsize=(12, 15))

# Histogram of Age in the first subplot
sns.histplot(df['Age'], ax=ax[0], color='blue', kde=True)
ax[0].set_title('Age Distribution')
ax[0].set_xlabel('Age')

# Histogram of Years of Experience in the second subplot
sns.histplot(df['Years of Experience'], ax=ax[1], color='orange', kde=True)
ax[1].set_title('Years of Experience Distribution')
ax[1].set_xlabel('Years of Experience')

# Histogram of Salary in the third subplot
sns.histplot(df['Salary'], ax=ax[2], color='green', kde=True)
ax[2].set_title('Salary Distribution')
ax[2].set_xlabel('Salary')

plt.tight_layout()
plt.show()


# In[16]:


from sklearn.preprocessing import LabelEncoder
categorical_cols = ['Gender', 'Education Level', 'Job Title']
le_dict={}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le


# In[17]:


import joblib
for col in categorical_cols:
    joblib.dump(le_dict[col], f"{col}_encoder.joblib")


# In[18]:


df


# In[19]:


X = df.drop('Salary', axis=1)
y = df['Salary']


# In[20]:


# Save column order for Streamlit
joblib.dump(X.columns.to_list(), "model_features.joblib")


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        'Model': name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R² Score': round(r2, 4)
    })

# Compare Results
results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
print("\nModel Performance Comparison:\n")
print(results_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('Model Comparison based on R² Score')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.xlabel('Model')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Actual vs Predicted for Best Model

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"Actual vs Predicted Salary ({best_model_name})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save best model 

joblib.dump(best_model, "best_modelofReg.joblib")
print(f" Best model ({best_model_name}) saved to best_modelofReg.joblib")


# In[ ]:





# In[ ]:




