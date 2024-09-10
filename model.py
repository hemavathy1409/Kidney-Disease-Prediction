import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Suppressing FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reading Dataset:
dataset = pd.read_csv('kidney_data1.csv')

# Dropping unnecessary feature:
dataset.drop('id', axis=1, inplace=True)

# Replacing Categorical Values with Numericals
replace_values = {'rbc': {'normal': 0, 'abnormal': 1},
                  'pc': {'normal': 0, 'abnormal': 1},
                  'pcc': {'notpresent': 0, 'present': 1},
                  'ba': {'notpresent': 0, 'present': 1},
                  'htn': {'yes': 1, 'no': 0},
                  'dm': {'yes': 1, 'no': 0},
                  'cad': {'yes': 1, 'no': 0},
                  'appet': {'good': 1, 'poor': 0, 'no': np.nan},
                  'pe': {'yes': 1, 'no': 0},
                  'ane': {'yes': 1, 'no': 0},
                  'classification': {'ckd\t': 'ckd'}
                  }

dataset.replace(replace_values, inplace=True)

# Converting Objective into Numericals:
dataset['eGFR'] = pd.to_numeric(dataset['eGFR'], errors='coerce')
dataset['eGFR'].fillna(dataset['eGFR'].median(), inplace=True)
dataset['eGFR'] = dataset['eGFR'].astype(int)
dataset['ckd_stage'] = dataset['ckd_stage'].astype('category')
dataset['ckd_stage'] = pd.cut(dataset['eGFR'],
                               bins=[0, 15, 30, 60, 90, np.inf],
                               labels=[5, 4, 3, 2, 1],
                               right=False).astype(int)

# Converting columns to numeric types
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
                'appet', 'pe', 'ane', 'eGFR', 'ckd_stage']

dataset[numeric_cols] = dataset[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Handling Missing Values:
# Impute missing values with mode for categorical variables and with median for numerical variables
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
    else:
        dataset[col] = dataset[col].fillna(dataset[col].median())

# Dropping feature (Multicollinearity):
dataset.drop('pcv', axis=1, inplace=True)

# Data Visualization:
plt.figure(figsize=(18, 6))

# Pairplot
plt.subplot(1, 2, 1)
sns.pairplot(dataset[['age', 'bp', 'sg', 'al', 'su', 'hemo', 'bgr', 'eGFR', 'classification']], hue='classification')
plt.title('Pairplot')

# Bar plot
plt.subplot(1, 2, 2)
# Count the frequency of each kidney disease stage
stage_counts = dataset['ckd_stage'].value_counts().sort_index()
# Define the stages of kidney disease
stage_labels = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
# Create a bar plot
sns.barplot(x=stage_labels, y=stage_counts, palette='viridis')
plt.title('Distribution of Kidney Disease Stages')
plt.xlabel('Kidney Disease Stage')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Independent and Dependent Features:
X = dataset[['age', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc', 'sg','bp','bgr','eGFR','sc','sod']]
y = dataset['classification']

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

# Model training and evaluation...
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)
y_pred = RandomForest.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Serialize the trained model
with open("tunned_kidney_Cancer_model.pkl", "wb") as file:
    pickle.dump(RandomForest, file)

# Hyperparameter tuning example...
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestClassifier()
tuned_model = RandomizedSearchCV(rf, param_grid, cv=5, n_iter=10)
tuned_model.fit(X_train, y_train)

# Printing Best Parameter during tuning
print(tuned_model.best_estimator_)

# Creating a pickle file for the classifier
filename = 'Kidney.pkl'
pickle.dump(RandomForest, open(filename, 'wb'))
