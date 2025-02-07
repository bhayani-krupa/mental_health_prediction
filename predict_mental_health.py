import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""**Uploading Dataset**"""

df = pd.read_csv('/content/survey.csv')
df.head()

df.shape

df.info()

"""**Checking for null values**"""

df.isnull().sum()

"""**Dropping irrelevant columns**"""

df.drop(['Country','state', 'Timestamp', 'comments'], axis = 1, inplace = True)

"""**Filling the null values**"""

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

df.isnull().sum()

"""**Finding unique values in columns 'Gender'**"""

value_counts = df["Gender"].value_counts()
value_counts

"""**Cleaning Gender column**"""

# Convert all gender values to lowercase to avoid case inconsistencies
df["Gender"] = df["Gender"].str.strip().str.lower()

# Replace specific terms to standardize
df["Gender"].replace(
    ["male", "m", "man", "msle", "make", "maile", "cis male", "guy", "mail", "mal",
     "male leaning androgynous", "male (cis)", "cis male", "malr"], "male", inplace=True
)

df["Gender"].replace(
    ["female", "f", "woman", "trans-female", "femake", "femail", "cis-female/femme",
     "cis female", "female (trans)", "cisfemalefemme", "woman", "female (cis)",
     "trans woman"], "female", inplace=True
)

df["Gender"].replace(
    ["neuter", "queer", "non-binary", "androgyne", "agender", "fluid", "enby",
     "genderqueer", "all", "nah", "something kinda male?", "a little about you",
     "guy (-ish) ^_^", "queer/she/they", "male-ish"], "other", inplace=True
)

df["Gender"].replace(
    ["cis man", "genderqueer", "p"], "other", inplace=True
)
df["Gender"] = df["Gender"].str.strip()

value_counts = df["Gender"].value_counts()
value_counts

"""**Cleaning Age Column**"""

df["Age"].value_counts().plot(kind = "bar", figsize = (10,8))

df.drop(df[(df["Age"]>60) | (df["Age"]<18)]. index, inplace = True)

import seaborn as sns
sns.distplot(df["Age"])
plt.title("Age Distribution")
plt.show()

"""**Visualizing the Dataset**"""

import seaborn as sns
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.show()

if "treatment" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="treatment", palette="Set2")
    plt.title("Treatment Seeking Distribution")
    plt.show()

if "treatment" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Gender", hue="treatment", palette="viridis")
    plt.title("Mental Health Treatment by Gender")
    plt.show()

plt.figure(figsize=(10, 40))
plt.subplot(9,2,1)
sns.countplot(x='self_employed', hue='treatment', data=df)
plt.title('Employment Type')
plt.show()

plt.figure(figsize=(10, 40))
plt.subplot(9,2,1)
sns.countplot(x='family_history', hue='treatment', data=df)
plt.title('Family History')
plt.show()

plt.figure(figsize=(10, 40))
plt.subplot(9,2,1)
sns.countplot(x='work_interfere', hue='treatment', data=df)
plt.title('Work Interference')
plt.show()

plt.figure(figsize=(10, 40))
plt.subplot(9,2,1)
sns.countplot(x='no_employees', hue='treatment', data=df)
plt.title('Number of Employees')
plt.show()

"""**EDA**"""

df.describe(include = 'all')

df.shape

import re

def normalize_text(text):
    """Normalize text by lowercasing and removing special characters."""
    if isinstance(text, str):
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

"""**Displaying categorical Data**"""

cat_data = df.select_dtypes(object)
cat_data.head()

"""**Converting Categorical Data into Numeric Value**"""

from sklearn.preprocessing import LabelEncoder
for col in cat_data:
    le = LabelEncoder()
    cat_data[col] = le.fit_transform(cat_data[col])
cat_data.head()

df.shape

from sklearn.preprocessing import LabelEncoder

for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train a simple Random Forest for feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Plot feature importances
importance_scores = model.feature_importances_
important_features = pd.Series(importance_scores, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
important_features.plot(kind="bar")
plt.title("Feature Importances")
plt.show()

# Select top features based on importance threshold
selected_features = important_features[important_features > 0.02].index.tolist()
X_selected = X[selected_features]
print(f"Selected Features: {selected_features}")

# Select features above importance threshold
selected_features = important_features[important_features > 0.02].index.tolist()
X_selected = X[selected_features]
print(f"Selected Features for Training: {selected_features}")

"""**Splitting the dataset into Training and Testing set**"""

X = df.drop("treatment", axis=1)
y = df["treatment"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=49)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""**Feature Scaling**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

"""**Training different models on the Training set**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=49),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=49)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

results_df = pd.DataFrame(results)
print("Model Comparison Results:")
print(results_df.sort_values(by='F1-score', ascending=False))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameters
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV
rf = RandomForestClassifier(random_state=49)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best Parameters & Accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# pip install shap

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

best_model = rf

best_model = grid_search.best_estimator_
explainer = shap.TreeExplainer(best_model)

explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train, 100))

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=49)
random_forest_model.fit(X_train, y_train)

import pickle
# open a file, where you ant to store the data
file = open('random_forest.pkl', 'wb')

# dump information to that file
pickle.dump(random_forest_model, file)

