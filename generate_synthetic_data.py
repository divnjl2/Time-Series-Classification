import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_complex_time_series(num_samples, label):
    """
    Generate a time series with more complex behaviors and increased overlap between classes.
    """
    time = np.linspace(0, 2 * np.pi, num_samples)
    choice = np.random.choice(['a', 'b', 'c'], p=[0.3, 0.4, 0.3]) if label == 0 else (
        np.random.choice(['a', 'b', 'c'], p=[0.2, 0.6, 0.2]) if label == 1 else
        np.random.choice(['a', 'b', 'c'], p=[0.4, 0.2, 0.4]))

    if choice == 'a':
        series = np.sin(time) + np.random.normal(0, 0.5, num_samples)
    elif choice == 'b':
        series = np.sin(2 * time) * np.linspace(0.5, 1.5, num_samples) + np.random.normal(0, 0.2, num_samples)
        if label != 0:
            spike_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
            series[spike_indices] += np.random.normal(0, 3, len(spike_indices))
    else:
        series = 0.5 * np.sin(time) + 0.5 * np.sin(3 * time + np.pi / 4) + np.random.normal(0, 0.1, num_samples)
        series += np.power(time, 2) / 50 if label == 2 else -np.power(time, 2) / 50

    return series

def generate_random_time_series(num_series_range, num_samples_range, num_classes):
    data = pd.DataFrame(columns=['series', 'label'])
    series_list = []
    num_series = np.random.randint(num_series_range[0], num_series_range[1] + 1)

    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)
        series = generate_complex_time_series(num_samples, label)
        series_list.append(pd.DataFrame({'series': [series.tolist()], 'label': [label]}))

    data = pd.concat(series_list, ignore_index=True)
    return data

def generate_and_save_datasets(num_datasets, num_series_range, num_samples_range, num_classes, base_filename='dataset'):
    for i in range(num_datasets):
        dataset = generate_random_time_series(num_series_range, num_samples_range, num_classes)
        filename = f"{base_filename}_{i}.csv"
        dataset.to_csv(filename, index=False)
        print(f"Saved {filename}")

def generate_large_dataset_v2(num_series, num_samples_range, num_classes, filename):
    series_list = []
    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples_per_series = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)  # Variable length
        series = generate_complex_time_series(num_samples_per_series, label)
        series_list.append({'series': series.tolist(), 'label': label})

    data = pd.DataFrame(series_list)
    data.to_csv(filename, index=False)
    print(f"Saved large dataset {filename}")


# Generate datasets with a variable number of series between 100 and 500, and variable series length between 30 and 100 samples
generate_and_save_datasets(40, (100, 500), (30, 100), 3)

# Generate large datasets with the new function
# Example call with updated parameters
generate_large_dataset_v2(100000, (30, 70), 3, 'large_dataset_series_v2.csv')


def generate_single_large_series(num_samples, num_classes, filename):
    """
    Generate a single large time series with a specified number of samples.
    Now introduces more variability in the series based on the label.
    """
    # Generate a sequence of labels for segments within the series
    segment_labels = np.random.randint(0, num_classes, size=num_samples // 1000)

    series = []
    for label in segment_labels:
        segment_length = 1000  # Each segment will be 1000 samples long
        segment = generate_complex_time_series(segment_length, label)
        series.extend(segment)

    # Ensure the series length matches num_samples
    series = series[:num_samples]

    # Convert the series into a DataFrame and save
    data = pd.DataFrame(
        {'series': [series], 'label': [segment_labels[0]]})  # Use the first segment's label as the overall label
    data.to_csv(filename, index=False)
    print(f"Saved single large series dataset {filename}")


# Example usage to generate a single series with 1,000,000 samples
generate_single_large_series(1000000, 3, 'single_large_series.csv')


"""Complex Time Series Generation (generate_complex_time_series)
Purpose: Generates a single time series with more complex behaviors, influenced by the input label. It introduces variability and overlaps between classes to mimic real-world complexities.
Time Series Creation:
Time axis: Generated using np.linspace(0, 2 * np.pi, num_samples).
Based on the label, a choice is made between three patterns ('a', 'b', 'c'), with different probabilities for each class.
Pattern 'a': A sine wave with added Gaussian noise.
Pattern 'b': A modulated sine wave (frequency doubled) with added noise and potential spikes for labels 1 and 2.
Pattern 'c': A combination of two sine waves with different frequencies, added Gaussian noise, and a quadratic trend that depends on the label.
Parameters:
num_samples: Number of points in the time series.
label: Determines the behavior and characteristics of the generated time series.
Random Time Series Collection Generation (generate_random_time_series)
Purpose: Generates a DataFrame containing a collection of randomly generated time series and their labels.
Time Series Collection Creation:
Randomly determines the number of series to generate within a specified range (num_series_range).
For each series, it selects a random label and number of samples, then generates the series using generate_complex_time_series.
Parameters:
num_series_range: Range for the number of time series to generate.
num_samples_range: Range for the length of each time series.
num_classes: Number of distinct labels or classes.
Dataset Saving Functions (generate_and_save_datasets, generate_large_dataset_v2)
Purpose: Generates multiple datasets or a single large dataset and saves them to CSV files.
Dataset Characteristics:
For generate_and_save_datasets, datasets with a variable number of series (between 100 and 500) and variable series lengths (between 30 and 100 samples) are created.
generate_large_dataset_v2 creates a dataset with 100,000 series, each with a variable length between 30 and 70 samples, to introduce more variability.
Single Large Series Generation (generate_single_large_series)
Purpose: Generates a single large time series composed of 1,000,000 samples, segmented by 1000 samples each with potentially different behaviors based on the segment's label.
Characteristics: This method introduces variability within the single series by varying the behavior of each segment according to a randomly assigned label.
2. Implementation and Evaluation of Synthetic Data (implementation)
Feature Extraction (extract_features)
Features Generated: For each time series, four features are extracted: mean, maximum, minimum, and standard deviation. These features are designed to capture the central tendency, dispersion, and range of the series.
Dataset Preprocessing and Evaluation (preprocess_and_evaluate)
Processes the loaded dataset, either treating it as a collection of individual series or a single large series.
For individual series, it directly extracts features from each series.
For the single large series, it segments the series into smaller parts, extracts features from each segment, and evaluates the model using cross-validation.
Uses a RidgeClassifierCV for classification, leveraging cross-validation for the single large series and a simple train-test split for individual series datasets."""