import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

df['popularity'] = pd.qcut(df['usersrated'], 3, labels=['low', 'medium', 'high'])

# relevant features
features = df[['numgeeklists', 'numwanting', 'siteviews']]

# Handle missing values
features.fillna(features.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split 
X_train, X_test, y_train, y_test = train_test_split(features_scaled, df['popularity'], test_size=0.2, random_state=42)

# Initialize and train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on the test set
y_pred = gnb.predict(X_test)

#  performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Add predictions to the original DataFrame
df['predicted_popularity'] = gnb.predict(scaler.transform(df[['numgeeklists', 'numwanting', 'siteviews']].fillna(features.mean())))


df.to_csv('classified_boardgames.csv', index=False)

print("Data with predicted popularity categories has been saved to 'classified_boardgames.csv'.")

