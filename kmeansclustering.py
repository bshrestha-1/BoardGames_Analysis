import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

data = df[['avgweight', 'maxplaytime']].dropna()

#  original data with indices 
original_data = df.loc[data.index].copy()

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split: 70% training, 20% testing, 10% validation
train_data, temp_data = train_test_split(data_scaled, test_size=0.3, random_state=42)
test_data, validation_data = train_test_split(temp_data, test_size=1/3, random_state=42)

# optimal number of clusters using Elbow Method
wcss = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_data)
    wcss.append(kmeans.inertia_)

# WCSS against k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

optimal_k = 4

# Train
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
final_kmeans.fit(train_data)

# Predict 
data['Cluster'] = final_kmeans.predict(data_scaled)

# Scatter plot with color-coded clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['avgweight'], y=data['maxplaytime'], hue='Cluster', palette='viridis', data=data)
plt.title(f'K-means Clustering with {optimal_k} Clusters')
plt.xlabel('Complexity (avgweight)')
plt.ylabel('Maximum Playtime (maxplaytime)')
plt.show()

# cluster labels to the original data
original_data['Cluster'] = data['Cluster']


original_data.to_csv('clustered_boardgames.csv', index=False)

#  model's efficiency and accuracy

# WCSS for the final model
final_wcss = final_kmeans.inertia_
print(f'WCSS for k={optimal_k}: {final_wcss}')

# Validation set accuracy: predicting cluster labels on validation data
validation_labels = final_kmeans.predict(validation_data)

