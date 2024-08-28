import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

# relevant features: avgweight for complexity and maxplaytime for playtime
features = df[['avgweight', 'maxplaytime']]

#  missing values 
features = features.dropna()

# Scale the features to normalize their scale
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# DBSCAN with an epsilon (eps) and minimum samples parameter
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(features_scaled)

# cluster labels
df['Cluster'] = clusters

# Outliers --> -1 
outliers = df[df['Cluster'] == -1]
inliers = df[df['Cluster'] != -1]


plt.figure(figsize=(10, 6))
plt.scatter(inliers['avgweight'], inliers['maxplaytime'], label='Inliers', alpha=0.7)
plt.scatter(outliers['avgweight'], outliers['maxplaytime'], color='red', label='Outliers', alpha=0.7)
plt.title('Anomaly Detection in Board Games')
plt.xlabel('Game Complexity (avgweight)')
plt.ylabel('Maximum Playtime (maxplaytime)')
plt.legend()
plt.show()

# Save the outliers 
outliers.to_csv('anomalous_games.csv', index=False)

