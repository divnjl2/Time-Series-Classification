import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_random_time_series(num_series, num_samples, num_classes):
    data = pd.DataFrame(columns=['series', 'label'])
    series_list = []

    for _ in range(num_series):
        # Random class selection
        label = np.random.randint(0, num_classes)

        # Generate series with structured variation
        if label == 0:
            series = np.random.randn(num_samples) + np.linspace(-1, 1, num_samples)
        elif label == 1:
            series = np.random.randn(num_samples) * np.linspace(1, 2, num_samples)
        else:
            series = np.random.randn(num_samples) * np.cos(np.linspace(0, 2*np.pi, num_samples))

        series_list.append(pd.DataFrame({'series': [series], 'label': [label]}))

    data = pd.concat(series_list, ignore_index=True)
    return data

# Generate the dataset
synthetic_dataset = generate_random_time_series(500, 50, 3)  # Adjust parameters as needed

# Check class distribution
class_distribution = synthetic_dataset['label'].value_counts()
print("Class Distribution:\n", class_distribution)

# Plot some sample series from each class
fig, axes = plt.subplots(3, 1, figsize=(10, 6))
for i in range(3):
    axes[i].plot(synthetic_dataset[synthetic_dataset['label'] == i].iloc[0]['series'])
    axes[i].set_title(f"Sample Series from Class {i}")
plt.tight_layout()
plt.show()
