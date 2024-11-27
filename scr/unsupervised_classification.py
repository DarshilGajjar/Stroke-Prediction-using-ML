# unsupervised_learning.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset file
data = pd.read_csv('stroke_data.csv')

# Select the relevant features for clustering
features = ['age', 'bmi', 'avg_glucose_level']
X = data[features]

# Perform KMeans clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Calculate the Silhouette and Calinski-Harabasz index
silhouette_avg = silhouette_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)

print(f'Silhouette Score: {silhouette_avg}')
print(f'Calinski-Harabasz Score: {calinski_harabasz}')

# 1D Histograms for each feature
for feature in features:
    plt.figure(figsize=(6, 4))
    plt.hist(X[feature], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'1D Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 3D Scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['age'], X['bmi'], X['avg_glucose_level'], c=labels, cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Avg Glucose Level')
ax.set_title('3D Scatter Plot of Clusters')
plt.show()