import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


file_path = 'boardgames.csv'
df = pd.read_csv(file_path)


print("Initial Data Info:")
print(df.info())
print("\nInitial Data Head:")
print(df.head())

# 1. Handle Missing Values
# numerical missing values with the median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# categorical missing values with 'Unknown'
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# 2. Remove Duplicates
df = df.drop_duplicates()

# 3. Normalize/Scale Numerical Data
# Standardize (mean=0, std=1)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 4. Feature Engineering
# New feature: Playtime Range
df['playtime_range'] = df['maxplaytime'] - df['minplaytime']

# Binning user ratings into categories
df['rating_category'] = pd.qcut(df['average'], 3, labels=['low', 'medium', 'high'])

# 5. categorical data to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['label', 'rating_category'])

# 6. Data Reduction
# drop columns not needed for analysis
df.drop(['description', 'gamelink'], axis=1, inplace=True)

# 7. Export Cleaned Data
df.to_csv('cleaned_boardgames.csv', index=False)

#  cleaned data info
print("Cleaned Data Info:")
print(df.info())
print("\nCleaned Data Head:")
print(df.head())

