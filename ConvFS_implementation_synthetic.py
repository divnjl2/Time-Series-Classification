import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
import time
from ast import literal_eval  # Import for safely evaluating strings as lists

# Assuming ConvFS_functions.py is in the same directory and correctly defined
from ConvFS_functions import generate_kernels, transform_and_select_features


def load_dataset(filename):
    df = pd.read_csv(filename)
    # Safely convert the 'series' column from string representations of lists to actual numpy arrays
    df['series'] = df['series'].apply(lambda x: np.array(literal_eval(x)))
    return df


def preprocess_and_evaluate(dataset):
    # Extract series and labels
    X_synthetic = np.array(dataset['series'].tolist())  # Convert list of arrays into a 2D numpy array directly
    y_synthetic = dataset['label'].values

    # Flatten each time series into a 1D array (if necessary)
    X_synthetic_flat = np.array([series.flatten() for series in X_synthetic])

    # Preprocessing
    avg_series_length = np.mean([len(x) for x in X_synthetic])

    # Generate kernels
    kernels = generate_kernels(50, 10000, int(avg_series_length))  # Assuming series_length is 50

    # Split synthetic data into train and test sets (example: 70% train, 30% test)
    n_samples = len(X_synthetic)
    train_size = int(0.7 * n_samples)
    X_train_flat = X_synthetic_flat[:train_size]
    y_train = y_synthetic[:train_size]
    X_test_flat = X_synthetic_flat[train_size:]
    y_test = y_synthetic[train_size:]

    # Transform and select features
    X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
        X_train_flat, kernels, y_train, is_train=True)

    X_test_transformed = transform_and_select_features(
        X_test_flat, kernels, selector=selector, scaler=scaler, is_train=False)

    # Classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

    # Train and test classifier
    classifier.fit(X_train_transformed, y_train)
    predictions = classifier.predict(X_test_transformed)
    accuracy = np.mean(predictions == y_test)

    return accuracy


def main():
    num_datasets = 10
    accuracies = []  # Collect accuracies for summary
    for i in range(num_datasets):
        filename = f"dataset_{i}.csv"
        dataset = load_dataset(filename)
        accuracy = preprocess_and_evaluate(dataset)
        accuracies.append(accuracy)
        print(f"Dataset {i} Accuracy: {accuracy:.4f}")

    # Optionally, print summary statistics
    print(f"\nSummary:\nAverage Accuracy: {np.mean(accuracies):.4f}\nStd Dev: {np.std(accuracies):.4f}")


if __name__ == "__main__":
    main()
