import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
from aeon.classification.convolution_based import Arsenal
from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
#from aeon.classification.deep_learning.fcn import FCNClassifier
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.deep_learning import IndividualInceptionClassifier
from aeon.classification.deep_learning.tapnet import TapNetClassifier
from aeon.classification.dictionary_based import BOSSEnsemble
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.classification.dictionary_based import IndividualBOSS
from aeon.classification.dictionary_based import MUSE
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from aeon.classification.dictionary_based import WEASEL
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.distance_based import ElasticEnsemble
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.distance_based import ShapeDTW
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.feature_based import MatrixProfileClassifier
from aeon.classification.interval_based import CanonicalIntervalForestClassifier
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.interval_based import SupervisedTimeSeriesForest
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.interval_based import RandomIntervalClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.classification.sklearn import ContinuousIntervalTree
from aeon.classification.sklearn import RotationForestClassifier
from keras_self_attention import SeqSelfAttention
import tensorflow_probability as tfp
import random

# Load the dataset
X, y, meta_data = load_classification("ElectricDevices")

# Extract dataset information
num_classes = len(meta_data['class_values'])
num_samples, num_features = X.shape[0], X.shape[1]

print(f"Dataset Information:")
print(f"Number of Classes: {num_classes}")
print(f"Number of Samples: {num_samples}")
print(f"Number of Features: {num_features}")

# Assuming 'X' is your time series dataset
print("Sample Lines:")
print(X[:5])  # Print the first 5 time series samples

# Assuming 'y' is your corresponding labels
print("Sample Labels:")
print(y[:5])  # Print the labels of the first 5 time series samples

# Assuming 'y' is your labels
num_labels = len(np.unique(y))
print(f"Number of Unique Labels: {num_labels}")

# Assuming 'y' is your labels
unique_labels = np.unique(y)
print("Unique Labels:")
print(unique_labels)

####################################################################
""" Plot one time series for each class of our dataset """
# Identify unique classes in your dataset
unique_classes = np.unique(y)

# Create a dictionary to store one time series example for each class
class_examples = {}

# Iterate through unique classes and find one example for each
for class_label in unique_classes:
    # Find the index of the first occurrence of the class in y
    index = np.where(y == class_label)[0][0]
    class_examples[class_label] = X[index]

# Plot the selected time series examples
plt.figure(figsize=(12, 6))
for class_label, example in class_examples.items():
    plt.plot(example.ravel(), label=f'Class {class_label}')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('One Time Series Example for Each Class')
plt.legend()
plt.show()
####################################################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of classifiers
classifiers = [
    Arsenal(),
    RocketClassifier(),
    CNNClassifier(),
    #FCNClassifier(),
    MLPClassifier(),
    InceptionTimeClassifier(),
    IndividualInceptionClassifier(),
    TapNetClassifier(),
    BOSSEnsemble(),
    ContractableBOSS(),
    IndividualBOSS(),
    MUSE(),
    TemporalDictionaryEnsemble(),
    WEASEL(),
    WEASEL_V2(),
    ElasticEnsemble(),
    KNeighborsTimeSeriesClassifier(),
    ShapeDTW(),
    Catch22Classifier(),
    MatrixProfileClassifier(),
    CanonicalIntervalForestClassifier(),
    DrCIFClassifier(),
    SupervisedTimeSeriesForest(),
    TimeSeriesForestClassifier(),
    RandomIntervalClassifier(),
    ShapeletTransformClassifier(),
    RDSTClassifier(),
    ContinuousIntervalTree(),
    RotationForestClassifier(),
]

# Initialize lists to store classifier names, accuracies, and confusion matrices
classifier_names = []
accuracies = []
confusion_matrices = []

# Loop through classifiers
for classifier in classifiers:
    classifier_name = type(classifier).__name__
    classifier_names.append(classifier_name)

    # Fit the classifier on the training data
    classifier.fit(X_train, y_train)

    # Predict labels on the test set
    predicted_labels = classifier.predict(X_test)

    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, predicted_labels)
    accuracies.append(accuracy)

    # Calculate and store the confusion matrix
    confusion = confusion_matrix(y_test, predicted_labels)
    confusion_matrices.append(confusion)

    # Print accuracy and classification report
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    report = classification_report(y_test, predicted_labels)
    print(f"{classifier_name} Classification Report:\n{report}")

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, accuracies, color='skyblue')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=90)
plt.show()

# Plot confusion matrices
for i, classifier_name in enumerate(classifier_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(len(meta_data['class_values'])), meta_data['class_values'], rotation=45)
    plt.yticks(np.arange(len(meta_data['class_values'])), meta_data['class_values'])
    plt.show()
