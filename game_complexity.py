import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

file_path = 'boardgames.csv'
df = pd.read_csv(file_path)

df = df.copy()

if 'boardgamemechanic_cnt' not in df.columns:
    df['boardgamemechanic_cnt'] = df['boardgamemechanic'].str.split(', ').apply(len)

features = df[['minplaytime', 'maxplaytime', 'minage', 'boardgamemechanic_cnt']]
target = df['avgweight']

# missing values
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict 
y_pred = rf.predict(X_test)

# performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Function to predict avgweight based on user input
def predict_avgweight(min_playtime, max_playtime, min_age, mechanic_count):
    
    input_data = pd.DataFrame([[min_playtime, max_playtime, min_age, mechanic_count]],
                              columns=['minplaytime', 'maxplaytime', 'minage', 'boardgamemechanic_cnt'])
   
    input_scaled = scaler.transform(input_data)
    predicted_weight = rf.predict(input_scaled)
    return predicted_weight[0]

# user input and predicting avgweight
min_playtime = int(input("Enter minimum playtime: "))
max_playtime = int(input("Enter maximum playtime: "))
min_age = int(input("Enter minimum age requirement: "))
mechanic_count = int(input("Enter number of mechanics: "))

predicted_weight = predict_avgweight(min_playtime, max_playtime, min_age, mechanic_count)
print(f'Predicted average weight for the board game is: {predicted_weight}')

