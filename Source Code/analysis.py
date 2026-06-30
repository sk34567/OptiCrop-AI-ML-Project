import pandas as pd
import numpy as np

# Display settings
pd.set_option('max_colwidth', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 8)

# Interactive shell settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Widgets
from ipywidgets import interact

# Machine Learning libraries
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(data.head())

# Univariate Analysis

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(2, 4, figsize=(18, 10))

sns.histplot(data['N'], kde=True, color='orange', ax=ax[0,0])
ax[0,0].set_title("Ratio of Nitrogen")

sns.histplot(data['P'], kde=True, color='blue', ax=ax[0,1])
ax[0,1].set_title("Ratio of Phosphorous")

sns.histplot(data['K'], kde=True, color='pink', ax=ax[0,2])
ax[0,2].set_title("Ratio of Potassium")

sns.histplot(data['temperature'], kde=True, color='green', ax=ax[0,3])
ax[0,3].set_title("Ratio of Temperature")

sns.histplot(data['humidity'], kde=True, color='purple', ax=ax[1,0])
ax[1,0].set_title("Ratio of Humidity")

sns.histplot(data['ph'], kde=True, color='red', ax=ax[1,1])
ax[1,1].set_title("Ratio of pH")

sns.histplot(data['rainfall'], kde=True, color='yellow', ax=ax[1,2])
ax[1,2].set_title("Ratio of Rainfall")

# Hide the last empty subplot
fig.delaxes(ax[1,3])

plt.suptitle("Distribution of Agricultural Conditions", fontsize=18)
plt.tight_layout()
plt.show()
# ==========================
# Bivariate Analysis
# ==========================

plt.figure(figsize=(8, 8))

sns.scatterplot(x=data['humidity'], y=data['label'])

plt.title("Humidity vs Crop Label")
plt.xlabel("Humidity")
plt.ylabel("Crop Label")

plt.show()
# ==========================
# Multivariate Analysis
# ==========================

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Count plot for all features
plt.figure(figsize=(10,6))
sns.countplot(data=data)

plt.title("Count Plot of Agricultural Features")
plt.xticks(rotation=45)

plt.show()
# ==========================
# Data Pre-processing
# Story 1: Check Dataset Information
# ==========================

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())
# ==========================
# Handling Outliers
# ==========================

plt.figure(figsize=(8,4))
sns.boxplot(data=data)

plt.title("Handling Outliers")
plt.show()

# IQR Method

Q1 = data['P'].quantile(0.25)
Q3 = data['P'].quantile(0.75)

IQR = Q3 - Q1

filter = (data['P'] >= Q1 - 1.5 * IQR) & (data['P'] <= Q3 + 1.5 * IQR)

data = data.loc[filter]

print("\nDataset Shape After Handling Outliers:")
print(data.shape)

# Log Transformation on Potassium

data['K'] = np.log1p(data['K'])

print("Log transformation applied on Potassium.")

# ==========================
# Extract Seasonal Crop Information
# ==========================

print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())

print("------------------------------------------------")

print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())

print("------------------------------------------------")

print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 50)]['label'].unique())

print("------------------------------------------------")

# ==========================
# Splitting Data into Train and Test Sets
# ==========================

# Features and Target
y = data['label']
x = data.drop('label', axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

print("\nThe shape of x train:", x_train.shape)
print("The shape of x test:", x_test.shape)
print("The shape of y train:", y_train.shape)
print("The shape of y test:", y_test.shape)

# ==========================
# K-Means Clustering
# ==========================

plt.rcParams['figure.figsize'] = (10,4)

wcss = []

for i in range(1,11):
    km = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )

    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.title("The Elbow Method", fontsize=20)
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

# Train K-Means Model

km = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)

y_means = km.fit_predict(x)

# Combine Cluster Results

a = data['label']

y_means = pd.DataFrame(y_means)

z = pd.concat([y_means, a], axis=1)

z = z.rename(columns={0:'cluster'})

print("\nLet's check the results after applying the K-Means Clustering Analysis\n")

print("Crops in First Cluster:", z[z['cluster']==0]['label'].unique())

print("----------------------------------------------------------")

print("Crops in Second Cluster:", z[z['cluster']==1]['label'].unique())

print("----------------------------------------------------------")

print("Crops in Third Cluster:", z[z['cluster']==2]['label'].unique())

print("----------------------------------------------------------")

print("Crops in Fourth Cluster:", z[z['cluster']==3]['label'].unique())

# ==========================
# Logistic Regression
# ==========================

model = LogisticRegression(max_iter=10000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\nLogistic Regression model trained successfully!")

# ==========================
# Evaluating Model Performance
# ==========================

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

# ==========================
# Save the Best Model
# ==========================

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as model.pkl")

# ==========================
# Predict the Best Crop
# ==========================

# Sample input values
N = 20
P = 20
K = 20

# Apply the same transformation used during training
K = np.log1p(K)

temperature = 35
humidity = 40
ph = 7
rainfall = 50

sample_data = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
)

prediction = model.predict(sample_data)

print("\nThe suggested crop for the given climatic condition is:", prediction[0])