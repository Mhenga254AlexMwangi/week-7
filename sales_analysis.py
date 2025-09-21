import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Task 1: Load and Explore the Dataset
print("First 5 rows of the dataset:")
print(df.head())
print()

print("Dataset structure:")
print(df.info())
print()

print("Missing values:")
print(df.isnull().sum())
print()

# Task 2: Basic Data Analysis
print("Basic statistics:")
print(df.describe())
print()

print("Mean values by species:")
species_means = df.groupby('species').mean()
print(species_means)
print()

# Task 3: Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Bar chart
species_means['sepal length (cm)'].plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Average Sepal Length by Species')
axes[0,1].set_ylabel('Sepal Length (cm)')

# Histogram
axes[1,0].hist(df['sepal length (cm)'], bins=15)
axes[1,0].set_title('Distribution of Sepal Length')
axes[1,0].set_xlabel('Sepal Length (cm)')
axes[1,0].set_ylabel('Frequency)
plt.tight_layout()
plt.savefig('analysis_results.png')
plt.show()

print("Analysis complete. Visualizations saved as 'analysis_results.png'")
