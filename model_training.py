# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import numpy as np

# Load the dataset
file_path = 'beer-servings.csv'
beer_data = pd.read_csv(file_path)

# Drop irrelevant columns
beer_data = beer_data.drop(['Unnamed: 0', 'country'], axis=1)

# Handle missing values
numeric_data = beer_data.select_dtypes(include=['number'])
beer_data[numeric_data.columns] = beer_data[numeric_data.columns].fillna(numeric_data.mean())

# One-hot encode the 'continent' column
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
continent_encoded = encoder.fit_transform(beer_data[['continent']])

# Create encoded DataFrame and concatenate it with the rest of the dataset
continent_df = pd.DataFrame(continent_encoded, columns=encoder.get_feature_names_out(['continent']))
beer_data = pd.concat([beer_data.drop('continent', axis=1), continent_df], axis=1)

# Features and target
X = beer_data.drop('beer_servings', axis=1)
y = beer_data['beer_servings']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
filename = 'beer_servings_model.pkl'
pickle.dump(model, open(filename, 'wb'))

print("Model trained and saved as beer_servings_model.pkl")
