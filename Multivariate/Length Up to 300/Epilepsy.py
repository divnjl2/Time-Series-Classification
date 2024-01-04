import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from sktime.datasets import load_UCR_UEA_dataset
import time
from sklearn.metrics import precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize


# Deep Learning:
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
#from sktime.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from aeon.classification.deep_learning.tapnet import TapNetClassifier


# Dictionary-based:
from aeon.classification.dictionary_based import (BOSSEnsemble, ContractableBOSS, IndividualBOSS,
                                                  TemporalDictionaryEnsemble, IndividualTDE, WEASEL, MUSE)


# Distance-based:
from aeon.classification.distance_based import ShapeDTW, ElasticEnsemble, KNeighborsTimeSeriesClassifier

# Feature-based:
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier

# Interval-based
from aeon.classification.interval_based import (CanonicalIntervalForestClassifier, DrCIFClassifier,
                                                SupervisedTimeSeriesForest, TimeSeriesForestClassifier)

# Kernel-based:
from aeon.classification.convolution_based import RocketClassifier, Arsenal

# Load the dataset
X, y = load_UCR_UEA_dataset("Epilepsy")

# Print classes and number of samples
unique_classes, class_counts = np.unique(y, return_counts=True)
print("Classes and Number of Samples:")
for class_label, count in zip(unique_classes, class_counts):
    print(f"Class {class_label}: {count} samples")

# Extract features using tslearn
X_processed = TimeSeriesScalerMinMax().fit_transform(to_time_series_dataset(X.iloc[:, 0]))
# Flatten each time series into a one-dimensional array
X_processed_flat = X_processed.reshape((X_processed.shape[0], -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_flat, y, test_size=0.2, random_state=42)

# Define a list of classifiers
classifiers = [
    MLPClassifier(n_epochs=100),
    CNNClassifier(n_epochs=100), #nai
    FCNClassifier(n_epochs=100),#nai
    MCDCNNClassifier(n_epochs=100), #nai
    # InceptionTimeClassifier(), #apeiros xronos
    # ResNetClassifier(), #oxi ME TIPOTA
    # TapNetClassifier(n_epochs=20,batch_size=4), #oxi

    # BOSSEnsemble(max_ensemble_size=3), #oxi
    ContractableBOSS(n_parameter_samples=10, max_ensemble_size=3), #nai
    IndividualBOSS(window_size=8, word_length=4, alphabet_size=6), #nai
    # TemporalDictionaryEnsemble(n_parameter_samples=250, max_ensemble_size=50, randomly_selected_params=50, random_state=47), #oxi apeirh wra
    IndividualTDE(), #nai
    WEASEL(support_probabilities=True),  # nai
    # MUSE(), #check error

    ShapeDTW(), #nai
    # ElasticEnsemble(proportion_of_param_options=0.1, proportion_train_for_test=0.1, distance_measures=["dtw", "ddtw"], majority_vote=True) #NO FUCKING WAY
    # KNeighborsTimeSeriesClassifier() #sad

    Catch22Classifier(), #nai
    FreshPRINCEClassifier(), #nai

    CanonicalIntervalForestClassifier(n_estimators=10, random_state=0), #nai
    DrCIFClassifier(n_estimators=10, random_state=0), #nai
    SupervisedTimeSeriesForest(), #nai
    TimeSeriesForestClassifier(), #nai

    RocketClassifier(), #nai
    Arsenal(), #nai
]

# Initialize lists to store classifier names, accuracies, and confusion matrices
classifier_names = []
accuracies = []
confusion_matrices = []
# Initialize lists to store results
execution_times = []
precisions = []
roc_auc_scores_macro = []
f1_scores = []


# Loop through classifiers
for classifier in classifiers:
    classifier_name = type(classifier).__name__
    classifier_names.append(classifier_name)

    # Measure the time taken for fitting
    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    # Predict labels on the test set
    predicted_labels = classifier.predict(X_test)

    # Calculate precision, accuracy, ROC-AUC score, and F1 score
    precision = precision_score(y_test, predicted_labels, average='weighted')
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average='weighted')

    # Store the results
    precisions.append(precision)
    accuracies.append(accuracy)
    f1_scores.append(f1_score_val)

    # Calculate ROC-AUC score
    if hasattr(classifier, "predict_proba"):
        # Convert multiclass labels to binary labels
        y_test_bin = np.zeros((y_test.shape[0], len(np.unique(y))))
        for i, class_label in enumerate(np.unique(y)):
            y_test_bin[:, i] = np.where(y_test == class_label, 1, 0)

        # Binarize the labels and calculate ROC-AUC for each class
        y_prob = classifier.predict_proba(X_test)
        n_classes = y_test_bin.shape[1]

        # Compute ROC curve and ROC-AUC score for each class
        roc_auc_scores_class = []
        num_plots = n_classes
        num_cols = 2  # Number of columns in subplots
        num_rows = -(-num_plots // num_cols)  # Ceiling division to calculate the number of rows

        plt.figure(figsize=(15, 5 * num_rows))

        for i in range(num_plots):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores_class.append(roc_auc)

            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{classifier_name} ROC-AUC Curves (Class {i})')
            plt.legend(loc="lower right")

        plt.tight_layout()
        plt.show()

        # Compute macro-average ROC-AUC score
        roc_auc_score_macro_val = np.mean(roc_auc_scores_class)
        roc_auc_scores_macro.append(roc_auc_score_macro_val)

        print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_score_macro_val:.2f}")

    # Print results
    print(f"{classifier_name} Execution Time: {execution_time:.2f}s")
    print(f"{classifier_name} Precision: {precision:.2f}")
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")

    # Print Classification Report
    report = classification_report(y_test, predicted_labels)
    print(f"{classifier_name} Classification Report:\n{report}")

    # Calculate and store the confusion matrix
    confusion = confusion_matrix(y_test, predicted_labels)
    confusion_matrices.append(confusion)


# Plot accuracies
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, accuracies, color='skyblue')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()


# Plot macro-average ROC-AUC scores
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, roc_auc_scores_macro, color='lightcoral')
plt.xlabel('Classifiers')
plt.ylabel('Macro-Average ROC-AUC Score')
plt.title('Classifier ROC-AUC Score Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Plot execution times
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, execution_times, color='lightgreen')
plt.xlabel('Classifiers')
plt.ylabel('Execution Time (s)')
plt.title('Classifier Execution Time Comparison')
plt.xticks(rotation=45)
plt.show()

# Plot precisions
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, precisions, color='gold')
plt.xlabel('Classifiers')
plt.ylabel('Precision')
plt.title('Classifier Precision Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Plot F1 scores
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, f1_scores, color='lightcoral')
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.title('Classifier F1 Score Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Plot confusion matrices together
num_classifiers = len(classifier_names)
num_cols = 5
num_rows = -(-num_classifiers // num_cols)  # Ceiling division to calculate the number of rows

plt.figure(figsize=(20, 4 * num_rows))  # Adjust the figure size based on the number of rows

for i, classifier_name in enumerate(classifier_names):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y), rotation=45)
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y))

plt.tight_layout()
plt.show()