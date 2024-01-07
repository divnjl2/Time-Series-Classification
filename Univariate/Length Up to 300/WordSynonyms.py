# Dataset: WordSynonyms, Dimensions: 1, Length:	275, Train Size: 267, Test Size: 638, Classes: 25

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.metrics import precision_score, f1_score, roc_auc_score
import time
from itertools import cycle
from sklearn.preprocessing import label_binarize


# Deep Learning:
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier

# Dictionary-based:
from aeon.classification.dictionary_based import (BOSSEnsemble, ContractableBOSS, IndividualBOSS,
                                                  TemporalDictionaryEnsemble, IndividualTDE, WEASEL, MUSE)

# Distance-based:
from aeon.classification.distance_based import ShapeDTW, KNeighborsTimeSeriesClassifier

# Feature-based:
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier

# Interval-based
from aeon.classification.interval_based import (CanonicalIntervalForestClassifier, DrCIFClassifier,
                                                SupervisedTimeSeriesForest, TimeSeriesForestClassifier)

# Kernel-based:
from aeon.classification.convolution_based import RocketClassifier, Arsenal

# Load the dataset
X_train_raw, y_train = load_UCR_UEA_dataset("WordSynonyms", split="train", return_X_y=True)
X_test_raw, y_test = load_UCR_UEA_dataset("WordSynonyms", split="test", return_X_y=True)


# Function to convert DataFrame to 2D numpy array
def dataframe_to_2darray(df):
    num_samples = df.shape[0]
    num_timesteps = len(df.iloc[0, 0])
    array_2d = np.empty((num_samples, num_timesteps))

    for i in range(num_samples):
        array_2d[i, :] = df.iloc[i, 0]

    return array_2d


# Convert and preprocess the data
scaler = TimeSeriesScalerMinMax()
X_train_processed = scaler.fit_transform(dataframe_to_2darray(X_train_raw))
X_test_processed = scaler.transform(dataframe_to_2darray(X_test_raw))  # Use the same scaler to transform test data

# Flatten each time series into a one-dimensional array for classifiers that require flat features
X_train_flat = X_train_processed.reshape((X_train_processed.shape[0], -1))
X_test_flat = X_test_processed.reshape((X_test_processed.shape[0], -1))


# Define a list of classifiers
classifiers = [MLPClassifier(), CNNClassifier(), FCNClassifier(), MCDCNNClassifier(),
               BOSSEnsemble(), ContractableBOSS(), IndividualBOSS(), TemporalDictionaryEnsemble(),
               IndividualTDE(), WEASEL(support_probabilities=True), MUSE(support_probabilities=True),
               ShapeDTW(), KNeighborsTimeSeriesClassifier(), Catch22Classifier(), FreshPRINCEClassifier(),
               SupervisedTimeSeriesForest(), TimeSeriesForestClassifier(),
               CanonicalIntervalForestClassifier(), DrCIFClassifier(), RocketClassifier(), Arsenal()]

# Initialize lists to store results
results = {"Classifier": [], "Execution Time": [], "Precision": [], "Accuracy": [],
           "F1 Score": [], "ROC-AUC Score (Macro)": [], "ROC-AUC Score (Micro)": [], "Confusion Matrix": []}

# Function to evaluate classifier
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    start_time = time.time()
    classifier.fit(X_train, y_train)
    execution_time = time.time() - start_time

    predicted_labels = classifier.predict(X_test)
    precision = precision_score(y_test, predicted_labels, average='weighted')
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average='weighted')
    confusion = confusion_matrix(y_test, predicted_labels)

    roc_auc_macro = roc_auc_micro = None
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        roc_auc_micro = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro')

    return execution_time, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion


# Preparing to plot ROC-AUC curves
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# Evaluate each classifier
for classifier in classifiers:
    classifier_name = type(classifier).__name__
    exec_time, precision, accuracy, f1_score_val, roc_auc_macro, roc_auc_micro, confusion = \
        evaluate_classifier(classifier, X_train_flat, X_test_flat, y_train, y_test)

    results["Classifier"].append(classifier_name)
    results["Execution Time"].append(exec_time)
    results["Precision"].append(precision)
    results["Accuracy"].append(accuracy)
    results["F1 Score"].append(f1_score_val)
    results["ROC-AUC Score (Macro)"].append(roc_auc_macro)
    results["ROC-AUC Score (Micro)"].append(roc_auc_micro)
    results["Confusion Matrix"].append(confusion)

    # Print results
    print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
    print(f"{classifier_name} Precision: {precision:.2f}")
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_macro:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Micro): {roc_auc_micro:.2f}")

    # Classification report
    start_time = time.time()
    predicted_labels = classifier.predict(X_test_flat)
    report = classification_report(y_test, predicted_labels)
    report_time = time.time() - start_time
    print(f"Classification report time: {report_time:.2f}s")
    print(f"{classifier_name} Classification Report:\n{report}")

    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test_flat)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_dict[classifier_name] = fpr
        tpr_dict[classifier_name] = tpr
        roc_auc_dict[classifier_name] = roc_auc

# Function to plot ROC-AUC curves in separate subplots
def plot_roc_auc_curves(fpr_dict, tpr_dict, roc_auc_dict, results, n_classes):
    num_classifiers = len(results["Classifier"])
    num_cols = 3  # for a two-column layout
    num_rows = np.ceil(num_classifiers / num_cols).astype(int)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for idx, classifier_name in enumerate(results["Classifier"]):
        for i in range(n_classes):
            axes[idx].plot(fpr_dict[classifier_name][i], tpr_dict[classifier_name][i], lw=2,
                           label=f'ROC curve of class {i} (area = {roc_auc_dict[classifier_name][i]:.2f})')
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'ROC-AUC for {classifier_name}')
        axes[idx].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

# Call the function to plot ROC-AUC curves
plot_roc_auc_curves(fpr_dict, tpr_dict, roc_auc_dict, results, n_classes)


# Plotting ROC-AUC curves
plt.figure(figsize=(15, 10))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
for classifier_name, color in zip(results["Classifier"], colors):
    for i in range(n_classes):
        plt.plot(fpr_dict[classifier_name][i], tpr_dict[classifier_name][i], color=color, lw=2,
                 label=f'ROC curve of class {i} for {classifier_name} (area = {roc_auc_dict[classifier_name][i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC-AUC curves for all classifiers')
plt.legend(loc="lower right")
plt.show()

# Function to plot results
def plot_results(results, metric, title, color):
    plt.figure(figsize=(10, 6))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel('Classifiers')
    plt.ylabel(metric)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=90, ha='right')
    plt.show()


# Plotting results with modified x-axis labels and dynamic y-axis limit for execution time
def plot_results_mod(results, metric, title, color, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel('Classifiers')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90, ha='right')  # This will prevent overlapping of names
    if metric == "Execution Time":
        max_execution_time = max(results[metric])
        plt.ylim(0, max_execution_time * 1.1)  # Add 10% headroom
    else:
        plt.ylim(0, 1)
    plt.show()

# Plotting results
plot_results(results, "Accuracy", "Classifier Accuracy Comparison", "skyblue")
plot_results(results, "ROC-AUC Score (Macro)", "Classifier Macro-Average ROC-AUC Score Comparison", "lightcoral")
plot_results_mod(results, "Execution Time", "Classifier Execution Time Comparison", "lightgreen", "Time (s)")
plot_results(results, "Precision", "Classifier Precision Comparison", "gold")
plot_results(results, "F1 Score", "Classifier F1 Score Comparison", "lightcoral")

# Plot confusion matrices together
num_classifiers = len(results["Classifier"])
num_cols = 3
num_rows = -(-num_classifiers // num_cols)  # Ceiling division

plt.figure(figsize=(20, 4 * num_rows))
for i, classifier_name in enumerate(results["Classifier"]):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(results["Confusion Matrix"][i], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion M. for {classifier_name}')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    tick_marks = np.arange(len(np.unique(y_train)))
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)
plt.tight_layout()
plt.show()