import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
import matplotlib.pyplot as plt
from ast import literal_eval

def load_dataset(filename):
    df = pd.read_csv(filename)
    df['series'] = df['series'].apply(lambda x: np.array(literal_eval(x)))
    return df

def extract_features(series):
    features = [np.mean(series), np.max(series), np.min(series), np.std(series)]
    return features

def plot_series(series, title):
    plt.figure(figsize=(10, 4))
    for s in series:
        plt.plot(s, alpha=0.5)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

def preprocess_and_evaluate(dataset, is_single_series=False):
    if len(dataset) == 0:
        print("Dataset is empty. Skipping.")
        return None

    if is_single_series:
        series = dataset['series'].iloc[0]
        # Generate synthetic features by segmenting the large series and extracting features from each segment
        segment_size = 100  # Define a reasonable segment size
        segments = [series[i:i+segment_size] for i in range(0, len(series), segment_size)]
        X_synthetic = [extract_features(segment) for segment in segments if len(segment) == segment_size]
        y_synthetic = [dataset['label'].iloc[0]] * len(X_synthetic)
    else:
        X_synthetic = [extract_features(series) for series in dataset['series']]
        y_synthetic = dataset['label'].values

    if len(X_synthetic) < 10:
        print("Not enough data for meaningful training and testing. Skipping.")
        return None

    X_synthetic = np.array(X_synthetic)
    y_synthetic = np.array(y_synthetic)

    # For a single large series, use cross-validation instead of a simple train-test split
    if is_single_series:
        from sklearn.model_selection import cross_val_score
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        scores = cross_val_score(classifier, X_synthetic, y_synthetic, cv=5)  # 5-fold cross-validation
        accuracy = np.mean(scores)
    else:
        split_index = int(len(X_synthetic) * 0.7)
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_synthetic[:split_index], y_synthetic[:split_index])
        predictions = classifier.predict(X_synthetic[split_index:])
        accuracy = np.mean(predictions == y_synthetic[split_index:])

    return accuracy



def main():
    num_standard_datasets = 40
    standard_accuracies = []
    for i in range(num_standard_datasets):
        filename = f"dataset_{i}.csv"
        dataset = load_dataset(filename)
        plot_series(dataset['series'][:10], f"Standard Dataset {i} Time Series")

        # Print dataset metrics
        num_samples = len(dataset)
        sample_lengths = [len(s) for s in dataset['series']]
        print(f"Standard Dataset {i}: Number of Samples = {num_samples}, Sample Lengths = {sample_lengths}")

        accuracy = preprocess_and_evaluate(dataset)
        if accuracy is not None:
            standard_accuracies.append(accuracy)
            print(f"Standard Dataset {i} Accuracy: {accuracy:.4f}, Number of Features: {len(extract_features(dataset['series'][0]))}")

    large_datasets = [
        'large_dataset_series_v2.csv',  # Dataset with 100,000 series of length 50
        'single_large_series.csv'       # Single series with 1,000,000 samples
    ]
    large_accuracies = []
    for filename in large_datasets:
        dataset = load_dataset(filename)
        is_single_series = filename == 'single_large_series.csv'

        if is_single_series:
            plot_series([dataset['series'].iloc[0]], f"Single Large Series")
            num_samples = 1
            sample_lengths = [len(dataset['series'].iloc[0])]
        else:
            plot_series(dataset['series'][:10], f"Large {filename} Time Series")
            num_samples = len(dataset)
            sample_lengths = [len(s) for s in dataset['series']]

        #print(f"{filename}: Number of Samples = {num_samples}, Sample Lengths = {sample_lengths}")

        accuracy = preprocess_and_evaluate(dataset, is_single_series=is_single_series)
        if accuracy is not None:
            large_accuracies.append(accuracy)
            # Assuming feature extraction method doesn't change, number of features is constant based on `extract_features`
            print(f"Large {filename} Accuracy: {accuracy:.4f}, Number of Features: {len(extract_features(dataset['series'][0]))}")
        else:
            print(f"Skipping {filename} due to insufficient data.")

    if standard_accuracies:
        print(f"\nSummary for Standard Datasets:\nAverage Accuracy: {np.mean(standard_accuracies):.4f}\nStd Dev: {np.std(standard_accuracies):.4f}")
    if large_accuracies:
        print(f"\nSummary for Large Datasets:\nAverage Accuracy: {np.mean(large_accuracies):.4f}\nStd Dev: {np.std(large_accuracies):.4f}")

if __name__ == "__main__":
    main()
