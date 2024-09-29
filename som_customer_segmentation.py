import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv(r"C:\Users\91984\Downloads\data\customer_segmentation.csv")

# Select features for segmentation
features = data[['Age', 'Income', 'SpendingScore']].values

# Normalize data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Create and train SOM
som = MiniSom(x=10, y=10, input_len=features.shape[1], sigma=1.0, learning_rate=0.5)
som.train_random(features, num_iteration=1000)

# Visualize results
plt.figure(figsize=(10, 10))
for i, x in enumerate(features):
    w = som.winner(x)
    plt.text(w[0], w[1], str(data.iloc[i]['CustomerID']), color='black')

plt.title('Self-Organizing Map of Customer Segments')
plt.savefig('output/som_output.png')
plt.show()
