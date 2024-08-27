import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
file_path = 'boardgames.csv'  # Replace with your actual file name
df = pd.read_csv(file_path)

# Ensure all required columns are numeric and match the exact column names
columns_needed = ['name', 'minplaytime', 'playerage', 'avgweight',
                  'boardgamemechanic_cnt', 'boardgamecategory_cnt',
                  'maxplaytime', 'minplayers', 'maxplayers', 'sortindex']

# Convert necessary columns to numeric, coerce errors to handle issues
for col in columns_needed[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in these columns
df = df.dropna(subset=columns_needed)

# Select the top 10 games based on 'sortindex'
top_10_games = df.nsmallest(10, 'sortindex')[columns_needed]  # 'sortindex' determines the ranking

# Function to create radar chart
def create_radar_chart(data, labels, title, color):
    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Repeat the first value to close the circle
    data += data[:1]
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.plot(angles, data, color=color, linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title(title, size=15, color=color, y=1.1)
    plt.show()

# Parameters to include in the radar chart
parameters = ['minplaytime', 'playerage', 'avgweight', 'boardgamemechanic_cnt',
              'boardgamecategory_cnt', 'maxplaytime', 'minplayers', 'maxplayers']

# Normalize data for radar chart
top_10_games_normalized = top_10_games.copy()
for param in parameters:
    top_10_games_normalized[param] = (top_10_games[param] - top_10_games[param].min()) / (top_10_games[param].max() - top_10_games[param].min())

# Generate radar charts for the top 10 games
colors = plt.cm.viridis(np.linspace(0, 1, 10))
for i in range(10):
    game_data = top_10_games_normalized.iloc[i][parameters].values.tolist()  # Select only the parameters
    game_name = top_10_games_normalized.iloc[i]['name']
    create_radar_chart(game_data, parameters, game_name, colors[i])

