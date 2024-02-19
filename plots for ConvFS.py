# RELATIVE ACCURACY PLOT
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


# Given accuracies for 1NN-DTW and ConvFS
dtw_accuracies = [0.65, 0.60, 0.62, 0.62, 0.65, 0.60, 0.62, 0.70, 0.66, 0.68, 0.58, 0.42, 0.76, 0.21, 0.31, 0.39, 0.98, 0.96, 0.71, 0.39, 0.91, 0.85, 0.81, 0.87, 0.75, 0.81, 0.09]
convfs_accuracies = [0.77, 0.75, 0.83, 0.67, 0.73, 0.67, 0.63, 0.85, 0.67, 0.69, 0.82, 0.65, 0.81, 0.68, 0.61, 0.46, 0.99, 0.99, 0.92, 0.98, 0.97, 0.99, 0.92, 0.99, 0.99, 0.95, 0.11]

# Now plot the results
plt.figure(figsize=(15, 10))
plt.scatter(dtw_accuracies, convfs_accuracies, color='blue')

# Add labels and title
plt.xlabel('1NN-DTW Accuracy')
plt.ylabel('ConvFS Accuracy')
plt.title('Relative accuracy of ConvFS versus 1NN-DTW')

# Diagonal line to show the equality of accuracies
plt.plot([0, 1], [0, 1], 'k--', label='Equal Accuracy Line')

# Calculate wins, draws, and losses
wins = sum(c > d for c, d in zip(convfs_accuracies, dtw_accuracies))
draws = sum(c == d for c, d in zip(convfs_accuracies, dtw_accuracies))
losses = sum(c < d for c, d in zip(convfs_accuracies, dtw_accuracies))

# Annotating the number of wins, draws and losses
plt.text(0.05, 0.9, f'ConvFS is better here\nW: {wins} | D: {draws} | L: {losses}', fontsize=9, verticalalignment='top')
plt.text(0.7, 0.1, '1NN-DTW is better here', fontsize=9)

# Set the limits for the axes
plt.xlim(0, 1)
plt.ylim(0, 1)

# Add a legend
plt.legend()

# Display the plot
plt.show()







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import cycle


# Train sizes and execution times (for simplicity, we'll use the execution times we've just calculated for the new dataset)
train_sizes = [36, 267, 450, 60, 390, 600, 20, 100, 3315]  # The last size is for PhonemeSpectra

# Now we need to create lists for execution times for each algorithm, including the estimated time for the PhonemeSpectra
# We will create a dictionary where the key is the algorithm and the value is the list of execution times
# We already have the estimated execution times for PhonemeSpectra, let's add the given execution times for the algorithms
exec_times_given = {
    "MLP": [69.04, 384.91, 964.90, 217.77, 504.00, 1092.67, 150.57, 248.87, 9225.54],
    "CNN": [51.16, 225.93, 396.95, 147.16, 249.61, 434.09, 81.54, 117.70, 5063.49],
    "FCN": [3019.23, 5168.09, 8046.22, 2538.76, 7081.59, 12986.30, 4519.68, 3327.11, 182429.19],
    "MCDCNN": [8.56, 18.22, 27.79, 9.68, 25.73, 57.72, 14.57, 19.41, 668.75],
    "TDE": [68.54, 497.04, 860.64, 130.35, 808.05, 2012.99, 33.13, 354.04, 8345.91],
    "iTDE": [2.37, 8.00, 13.30, 2.72, 9.98, 21.55, 6.10, 2.89, 234.11],
    "MUSE": [4.81, 48.73, 70.03, 12.35, 51.97, 62.04, 27.02, 8.48, 1020.76],
    "KNN": [98, 503.92, 668.35, 78.16, 626.52, 1131.91, 399.69, 100, 13443.98],
    "catch22": [2.39, 3.19, 3.64, 2.69, 3.25, 7.65, 3.72, 2.17, 149.19],
    "FreshPRINCE": [73.47, 350.73, 637.87, 137.77, 328.90, 1724.42, 731.92, 175.97, 20362.86],
    "STSF": [9.33, 72.00, 149.23, 20.27, 76.59, 360.76, 12.11, 34.44, 1220.68],
    "TSF": [6.08, 25.94, 40.89, 7.87, 23.56, 85.41, 8.96, 12.35, 523.08],
    "CIF": [144.31, 654.75, 632.62, 227.71, 579.96, 1394.48, 432.57, 402.52, 17041.90],
    "DrCIF": [235.51, 1273.10, 1754.03, 220.22, 975.43, 2166.12, 523.23, 368.33, 22722.00],
    "ROCKET": [8.94, 35.98, 34.08, 10.27, 31.00, 90.79, 26.79, 13.52, 967.77],
    "Arsenal": [36.77, 177.93, 173.46, 46.82, 150.31, 447.12, 132.12, 62.32, 4646]}

# Sort the train sizes and corresponding execution times for each algorithm
sorted_indices = np.argsort(train_sizes)
sorted_train_sizes = np.array(train_sizes)[sorted_indices]
convfs_times_new = [2, 7.2, 11.8, 3, 9.9, 23, 4, 3.1, 443.5]  # Updated ConvFS times for each dataset including PhonemeSpectra

# Sort the execution times according to the sorted train sizes
sorted_exec_times = {algo: np.array(times)[sorted_indices] for algo, times in exec_times_given.items()}
convfs_times_sorted_new = np.array(convfs_times_new)[sorted_indices]

# Define the new colors
color_names = [
    'midnightblue', 'indianred', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan',
    'mediumaquamarine', 'chocolate', 'palegreen', 'antiquewhite', 'tan', 'darkseagreen', 'aquamarine',
    'cadetblue', 'powderblue', 'thistle', 'palevioletred'
]
colors = cycle(color_names)

# Plotting
plt.figure(figsize=(10, 7))

# Plot the algorithms with the new colors
for (algo, times), color in zip(sorted_exec_times.items(), colors):
    # Apply lowess smoothing
    smooth_data = lowess(times, sorted_train_sizes, frac=0.3, it=0, return_sorted=True)
    smooth_x, smooth_y = smooth_data[:, 0], smooth_data[:, 1]
    plt.plot(smooth_x, smooth_y, label=algo, color=color)


# Adding the updated ConvFS with the new times
convfs_smooth_data_new = lowess(convfs_times_sorted_new, sorted_train_sizes, frac=0.3, it=0, return_sorted=True)
convfs_smooth_x_new, convfs_smooth_y_new = convfs_smooth_data_new[:, 0], convfs_smooth_data_new[:, 1]

# Plot the updated ConvFS with a black thick line
plt.plot(convfs_smooth_x_new, convfs_smooth_y_new, label='ConvFS', color='black', linewidth=2.5)

# Use log scale on both axes
plt.xscale('log', base=2)
plt.yscale('log')

# Formatting the axes
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# Axis labels and title
plt.xlabel('Train Set Size')
plt.ylabel('Time (s)')
plt.title('Time vs Train Set Size')

# Grid and legend
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()

# Show the plot
plt.show()































import matplotlib.pyplot as plt
import numpy as np

# Data from the table
classifiers = [
    "MLP", "CNN", "FCN", "MCDCNN", "TDE", "iTDE",
     "MUSE", "KNN", "catch22", "Fr.PRINCE", "STSF", "TSF",
    "CIF", "DrCIF", "ROCKET", "Arsenal"
]
accuracies = [
    0.4633, 0.4833, 0.4733, 0.5067, 0.51, 0.24, 0.3633, 0.4433, 0.4267, 0.5033, 0.51, 0.5233, 0.5333, 0.51, 0.594, 0.57
]
convfs_accuracy = 0.59 # mean

# Calculate improvements
improvements = [convfs_accuracy - acc for acc in accuracies]



# Plotting all improvements in one plot for better comparison
fig, ax = plt.subplots(figsize=(14, 8))

# Sorting classifiers by improvement for better visualization
sorted_indices = np.argsort(improvements)
sorted_classifiers = [classifiers[i] for i in sorted_indices]
sorted_improvements = [improvements[i] for i in sorted_indices]

bars = ax.bar(sorted_classifiers, sorted_improvements, color='skyblue')
ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracy Improvement')
ax.set_title('Accuracy Improvement of ConvFS over Other Classifiers')
ax.axhline(0, color='black', linewidth=0.8)  # Add a line at 0 improvement for reference
ax.set_xticklabels(sorted_classifiers, rotation=45, ha='right')

# Annotating the improvement values on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()



























import matplotlib.pyplot as plt

# Define the time points for the plot
time_points = list(range(1, 101))

# K-Fold Cross-Validation (Example with 5 folds)
k_folds = [
    ([*range(1, 21)], [*range(21, 41)]),
    ([*range(21, 41)], [*range(41, 61)]),
    ([*range(41, 61)], [*range(61, 81)]),
    ([*range(61, 81)], [*range(81, 101)]),
    ([*range(81, 101)], [*range(1, 21)]),
]

# Time Series Split (Example with 5 splits)
ts_splits = [
    ([*range(1, 21)], [*range(21, 41)]),
    ([*range(1, 41)], [*range(41, 61)]),
    ([*range(1, 61)], [*range(61, 81)]),
    ([*range(1, 81)], [*range(81, 101)]),
    ([*range(1, 101)], []),
]

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Customize font sizes
plt.rcParams.update({'font.size': 16})  # Adjusting the font size globally

# Plot K-Fold Cross-Validation
for i, (train, test) in enumerate(k_folds):
    ax1.scatter(train, [i + 1] * len(train), color="blue", label="Training Data" if i == 0 else "")
    ax1.scatter(test, [i + 1] * len(test), color="red", label="Testing Data" if i == 0 else "")
ax1.set_title("K-Fold Cross-Validation")
ax1.set_xlabel("Time Points")
ax1.set_ylabel("Fold")
ax1.legend()

# Plot Time Series Split
for i, (train, test) in enumerate(ts_splits):
    ax2.scatter(train, [i + 1] * len(train), color="blue", label="Training Data" if i == 0 else "")
    ax2.scatter(test, [i + 1] * len(test), color="red", label="Testing Data" if i == 0 else "")
ax2.set_title("Time Series Split")
ax2.set_xlabel("Time Points")
ax2.set_ylabel("Split")
ax2.legend()

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Data for the ConvFS algorithm
convfs_fpr = np.array([0., 0.00819672, 0.00943396, 0.01639344, 0.01886792, 0.02459016,
                       0.02830189, 0.03278689, 0.03773585, 0.04098361, 0.04716981, 0.04918033,
                       0.05660377, 0.05737705, 0.06557377, 0.06603774, 0.07377049, 0.0754717,
                       0.08196721, 0.08490566, 0.09016393, 0.09433962, 0.09836066, 0.10655738,
                       0.11320755, 0.12295082, 0.13114754, 0.13934426, 0.14150943, 0.1509434,
                       0.1557377, 0.18032787, 0.19672131, 0.19811321, 0.21311475, 0.22131148,
                       0.22641509, 0.23584906, 0.24590164, 0.29245283, 0.29508197, 0.32786885,
                       0.36065574, 0.40163934, 0.42622951, 0.50943396, 0.52830189, 0.6147541,
                       0.70491803, 0.8852459, 0.90163934, 0.99056604, 1.])
convfs_tpr = np.array([0.1572327, 0.18238994, 0.25968462, 0.38547079, 0.41445629, 0.45219214,
                       0.50533224, 0.5493574, 0.56868107, 0.62528484, 0.63011576, 0.63640507,
                       0.65572874, 0.67459666, 0.69346459, 0.70795734, 0.72682527, 0.76064169,
                       0.766931, 0.77659284, 0.79546076, 0.80029168, 0.80658099, 0.81287029,
                       0.81770121, 0.82399052, 0.83027983, 0.83656914, 0.84140005, 0.84623097,
                       0.85252028, 0.8650989, 0.87138821, 0.87621912, 0.88250843, 0.88879774,
                       0.89362866, 0.89845958, 0.90474888, 0.91924164, 0.93182025, 0.93810956,
                       0.94439887, 0.95068818, 0.95697749, 0.95697749, 0.95697749, 0.96326679,
                       0.9695561, 0.9695561, 0.9695561, 0.97438702, 1.])

# Load the existing plot image to overlay the new curve
img = plt.imread(r'C:\Users\sophi\PycharmProjects\TimeSeriesClassification\Ouput Plots\Univariate Plots\ArrowHead_macro_average_roc_curve.png')
# For now, we'll make a simple linear transformation to fit the curve into the estimated plot area.
# We'll scale the FPR and TPR values from the range [0, 1] to [0.02, 0.98] for FPR and [0, 1] for TPR.
convfs_fpr_adjusted = convfs_fpr * 0.9
convfs_tpr_adjusted = convfs_tpr * 0.9

# Plotting the existing image
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')

# Adding the ConvFS ROC curve with a thick black line, adjusted to fit within the plot
ax.plot(convfs_fpr_adjusted, convfs_tpr_adjusted, label='macro-average ROC curve of ConvFS (area = 0.907)', color='black', linewidth=2)

# Disable the axis
ax.axis('off')

plt.show()