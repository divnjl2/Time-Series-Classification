import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_complex_time_series(num_samples, label):
    """
    Generate a time series with more complex behaviors and increased overlap between classes.
    """
    time = np.linspace(0, 2 * np.pi, num_samples)
    # Introduce randomness in the selection process for generating series
    choice = np.random.choice(['a', 'b', 'c'], p=[0.3, 0.4, 0.3]) if label == 0 else (
        np.random.choice(['a', 'b', 'c'], p=[0.2, 0.6, 0.2]) if label == 1 else
        np.random.choice(['a', 'b', 'c'], p=[0.4, 0.2, 0.4]))

    if choice == 'a':
        series = np.sin(time) + np.random.normal(0, 0.5, num_samples)
    elif choice == 'b':
        series = np.sin(2 * time) * np.linspace(0.5, 1.5, num_samples) + np.random.normal(0, 0.2, num_samples)
        if label != 0:  # Add spikes for non-zero labels to introduce variability
            spike_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
            series[spike_indices] += np.random.normal(0, 3, len(spike_indices))
    else:
        series = 0.5 * np.sin(time) + 0.5 * np.sin(3 * time + np.pi / 4) + np.random.normal(0, 0.1, num_samples)
        series += np.power(time, 2) / 50 if label == 2 else -np.power(time, 2) / 50  # Vary trend based on label

    return series  # This line returns the generated series


def generate_random_time_series(num_series_range, num_samples_range, num_classes):
    data = pd.DataFrame(columns=['series', 'label'])
    series_list = []

    # Randomly determine the number of series to generate within the specified range
    num_series = np.random.randint(num_series_range[0], num_series_range[1] + 1)

    for _ in range(num_series):
        label = np.random.randint(0, num_classes)

        # Randomly determine the number of samples for this series within the specified range
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



# Example usage
# Generate datasets with a variable number of series between 100 and 500,
# and variable series length between 30 and 100 samples
generate_and_save_datasets(10, (100, 500), (30, 100), 3)



def generate_large_dataset(num_series, num_samples, num_classes, filename):
    """Generate a large dataset with specified series count and sample length."""
    series_list = []
    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        series = generate_complex_time_series(num_samples, label)
        series_list.append({'series': series.tolist(), 'label': label})

    # Convert to DataFrame
    data = pd.DataFrame(series_list)
    data.to_csv(filename, index=False)
    print(f"Saved large dataset {filename}")

# Example usage for a single large dataset with 1,000,000 samples or series length
# generate_large_dataset(1, 1000000, 3, 'large_dataset_samples.csv') # For 1,000,000 samples
# generate_large_dataset(1000000, 50, 3, 'large_dataset_series.csv') # For 1,000,000 series of length 50
generate_large_dataset(1, 1000000, 3, 'large_dataset_samples.csv')  # For 1,000,000 samples
generate_large_dataset(1000000, 50, 3, 'large_dataset_series.csv')  # For 1,000,000 series
