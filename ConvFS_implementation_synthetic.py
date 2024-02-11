import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from ConvFS_functions import generate_kernels, transform_and_select_features
import time
from generate_synthetic_data import generate_random_time_series

# Generate synthetic data
n_samples = 500
series_length = 50
num_classes = 3  # Specify the number of classes
synthetic_data = generate_random_time_series(n_samples, series_length, num_classes)

# Extract series and labels
X_synthetic = list(synthetic_data['series'])
y_synthetic = synthetic_data['label'].values

# Flatten each time series into a 1D array
X_synthetic_flat = np.array([series.flatten() for series in X_synthetic])

# Preprocessing
avg_series_length = np.mean([len(x) for x in X_synthetic])

# Generate kernels
kernels = generate_kernels(series_length, 10000, int(avg_series_length))

# Split synthetic data into train and test sets (example: 70% train, 30% test)
train_size = int(0.7 * n_samples)
X_train_flat = X_synthetic_flat[:train_size]
y_train = y_synthetic[:train_size]
X_test_flat = X_synthetic_flat[train_size:]
y_test = y_synthetic[train_size:]

# Transform and select features for training set
X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
    X_train_flat, kernels, y_train, is_train=True)

# Transform and select features for test set using the same selector
X_test_transformed = transform_and_select_features(
    X_test_flat, kernels, selector=selector, scaler=scaler, is_train=False)

# Classifier
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

# Train classifier
start_time = time.time()
classifier.fit(X_train_transformed, y_train)
training_time = time.time() - start_time

# Test classifier
start_time = time.time()
predictions = classifier.predict(X_test_transformed)
test_time = time.time() - start_time
accuracy = np.mean(predictions == y_test)

# Print results
print(f"Synthetic Dataset Accuracy: {accuracy}")
print(f"Training Time: {training_time}s, Test Time: {test_time}s")
