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


def extract_features(series):
    # Example feature extraction: mean, max, min, standard deviation
    features = [np.mean(series), np.max(series), np.min(series), np.std(series)]
    return features

def preprocess_and_evaluate(dataset):
    if len(dataset) == 0:
        print("Dataset is empty. Skipping.")
        return None

    # Extract series and labels
    X_synthetic = [extract_features(series) for series in dataset['series']]
    y_synthetic = dataset['label'].values

    if len(X_synthetic) < 10:  # Arbitrary low number to ensure some data is present
        print("Not enough data for meaningful training and testing. Skipping.")
        return None

    # Convert list of feature vectors into a 2D numpy array
    X_synthetic = np.array(X_synthetic)

    # Proceed with training and evaluation
    split_index = int(len(X_synthetic) * 0.7)
    if split_index == 0:
        print("Insufficient data after split. Skipping.")
        return None

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_synthetic[:split_index], y_synthetic[:split_index])
    predictions = classifier.predict(X_synthetic[split_index:])
    accuracy = np.mean(predictions == y_synthetic[split_index:])

    return accuracy


def main():
    # Handle standard datasets
    num_standard_datasets = 10  # Number of standard datasets
    standard_accuracies = []  # Collect accuracies for standard datasets
    for i in range(num_standard_datasets):
        filename = f"dataset_{i}.csv"
        dataset = load_dataset(filename)
        accuracy = preprocess_and_evaluate(dataset)
        if accuracy is not None:  # Only append and print if accuracy is not None
            standard_accuracies.append(accuracy)
            print(f"Standard Dataset {i} Accuracy: {accuracy:.4f}")

    # Handle large datasets
    large_datasets = [
        'large_dataset_samples.csv',  # Dataset with 1,000,000 samples
        'large_dataset_series.csv'    # Dataset with 1,000,000 series
    ]
    large_accuracies = []  # Collect accuracies for large datasets
    for filename in large_datasets:
        dataset = load_dataset(filename)
        accuracy = preprocess_and_evaluate(dataset)
        if accuracy is not None:  # Check if accuracy is not None before printing
            large_accuracies.append(accuracy)
            print(f"Large {filename} Accuracy: {accuracy:.4f}")
        else:
            print(f"Skipping {filename} due to insufficient data.")

    # Print summary statistics for both standard and large datasets
    if standard_accuracies:
        print(f"\nSummary for Standard Datasets:\nAverage Accuracy: {np.mean(standard_accuracies):.4f}\nStd Dev: {np.std(standard_accuracies):.4f}")
    if large_accuracies:
        print(f"\nSummary for Large Datasets:\nAverage Accuracy: {np.mean(large_accuracies):.4f}\nStd Dev: {np.std(large_accuracies):.4f}")




if __name__ == "__main__":
    main()
