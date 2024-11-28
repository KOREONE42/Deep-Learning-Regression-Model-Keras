import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Create a synthetic dataset (or load your own dataset)
np.random.seed(0)
num_samples = 1000
num_features = 8

# Generate features and target variable
X = np.random.rand(num_samples, num_features)
y = X @ np.random.rand(num_features) + np.random.normal(0, 0.1, num_samples)

# Normalize the data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Apply normalization
X_normalized = (X - mean) / std

# Store mean squared errors for each iteration
mse_list_normalized_100_epochs = []

# Repeat the training and evaluation 50 times for normalized data
for _ in range(50):
    # Split the normalized data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=None)
    
    # Create the model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(num_features,)))  # Hidden layer
    model.add(Dense(1))  # Output layer for regression
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model for 100 epochs (changed from 50 to 100)
    model.fit(X_train, y_train, epochs=100, verbose=0)  # Train without verbose output
    
    # Evaluate the model
    y_pred = model.predict(X_test)  # Generate predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
    mse_list_normalized_100_epochs.append(mse)  # Store the MSE

# Calculate mean and standard deviation of the MSEs
mean_mse_normalized_100_epochs = np.mean(mse_list_normalized_100_epochs)
std_mse_normalized_100_epochs = np.std(mse_list_normalized_100_epochs)

# Print results for normalized data with 100 epochs
print(f'Mean Normalized MSE (100 epochs): {mean_mse_normalized_100_epochs:.4f}')
print(f'Standard Deviation of Normalized MSE (100 epochs): {std_mse_normalized_100_epochs:.4f}')

# You can also compare with the previous mean MSE from Step B
# Replace mean_mse_normalized with the mean value from Part B
# print(f'Comparison with previous mean MSE (50 epochs): {mean_mse_normalized:.4f}')