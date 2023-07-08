import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

# Select relevant features: avgweight for complexity and maxplaytime for playtime
features = df[['avgweight', 'maxplaytime']]

# Handle missing values if necessary
features = features.dropna()

# Scale the features to normalize their scale
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize DBSCAN with an epsilon (eps) and minimum samples parameter
# These parameters may need tuning depending on the density of your data
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(features_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = clusters

# Outliers are typically labeled as -1 in the clusters produced by DBSCAN
outliers = df[df['Cluster'] == -1]
inliers = df[df['Cluster'] != -1]

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(inliers['avgweight'], inliers['maxplaytime'], label='Inliers', alpha=0.7)
plt.scatter(outliers['avgweight'], outliers['maxplaytime'], color='red', label='Outliers', alpha=0.7)
plt.title('Anomaly Detection in Board Games')
plt.xlabel('Game Complexity (avgweight)')
plt.ylabel('Maximum Playtime (maxplaytime)')
plt.legend()
plt.show()

# Save the outliers to a CSV file for further analysis or review
outliers.to_csv('anomalous_games.csv', index=False)

