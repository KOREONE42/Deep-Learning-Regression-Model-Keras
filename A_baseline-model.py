import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Create a synthetic dataset
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
mse_list_normalized = []

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
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, verbose=0)  # Train without verbose output
    
    # Evaluate the model
    y_pred = model.predict(X_test)  # Generate predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
    mse_list_normalized.append(mse)  # Store the MSE

# Calculate mean and standard deviation of the MSEs
mean_mse_normalized = np.mean(mse_list_normalized)
std_mse_normalized = np.std(mse_list_normalized)

# Print results for normalized data
print(f'Mean Normalized MSE: {mean_mse_normalized:.4f}')
print(f'Standard Deviation of Normalized MSE: {std_mse_normalized:.4f}')

# Compare with previous results (Assuming previously stored mean_mse)
# Replace mean_mse with the mean MSE from your previous part A run
# mean_mse_previous = ...  # The mean value from Step A
# print(f'Comparison with previous mean MSE: {mean_mse_previous:.4f}')