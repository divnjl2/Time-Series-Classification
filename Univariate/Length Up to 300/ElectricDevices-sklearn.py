import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sktime.datasets import load_UCR_UEA_dataset

# Load the dataset
X, y = load_UCR_UEA_dataset("ElectricDevices")

# Assuming 'X' is your time series dataset
print("Sample Lines:")
print(X[:5])  # Print the first 5 time series samples

# Assuming 'y' is your corresponding labels
print("Sample Labels:")
print(y[:5])  # Print the labels of the first 5 time series samples

# Assuming 'y' is your labels
num_labels = len(np.unique(y))
print(f"Number of Unique Labels: {num_labels}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of classifiers
classifiers = [
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    MLPClassifier(),
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
plt.xticks(rotation=45)
plt.show()

# Plot confusion matrices
for i, classifier_name in enumerate(classifier_names):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y), rotation=45)
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.show()
