import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from sktime.datasets import load_UCR_UEA_dataset
import time
from sklearn.metrics import precision_score, roc_auc_score, f1_score, roc_curve, auc

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression

# Deep Learning:
from sktime.classification.deep_learning.mlp import MLPClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.macnn import MACNNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
#from aeon.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
#from aeon.classification.deep_learning.tapnet import TapNetClassifier


# Dictionary-based:
from aeon.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import IndividualBOSS
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.dictionary_based import IndividualTDE
from aeon.classification.dictionary_based import WEASEL
from sktime.classification.dictionary_based import MUSE

# Distance-based:
from aeon.classification.distance_based import ShapeDTW, ElasticEnsemble
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier as AeonKNeighborsTimeSeriesClassifier

# Feature-based:
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier

# Interval-based
from aeon.classification.interval_based import CanonicalIntervalForestClassifier, DrCIFClassifier, SupervisedTimeSeriesForest, TimeSeriesForestClassifier

# Kernel-based:
from aeon.classification.convolution_based import RocketClassifier, Arsenal

# Load the dataset
X, y = load_UCR_UEA_dataset("ElectricDevices")
# Extract features using tslearn
X_processed = TimeSeriesScalerMinMax().fit_transform(to_time_series_dataset(X.iloc[:, 0]))
# Flatten each time series into a one-dimensional array
X_processed_flat = X_processed.reshape((X_processed.shape[0], -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed_flat, y, test_size=0.2, random_state=42)

# Define a list of classifiers
classifiers = [
    # RandomForestClassifier(),
    # LogisticRegression(),

    # MLPClassifier(n_epochs=100),
    # CNNClassifier(n_epochs=100), #nai
    # FCNClassifier(n_epochs=100),#oxi
    # MACNNClassifier(n_epochs=100), #oxi
    # MCDCNNClassifier(n_epochs=100), #nai
    # InceptionTimeClassifier(),
    # ResNetClassifier(n_epochs=100), #oxi
    # TapNetClassifier(), #oxi

    # BOSSEnsemble(feature_selection='chi2'), #oxi
    # ContractableBOSS(),
    # IndividualBOSS(),
    # TemporalDictionaryEnsemble(),
    # IndividualTDE(),
    # WEASEL(support_probabilities=True),  # nai
    # MUSE(),

    # ShapeDTW(), #nai
    # ElasticEnsemble(), #oxi - pollh wra
    # AeonKNeighborsTimeSeriesClassifier(), #polli wra

    # Catch22Classifier(), #nai
    FreshPRINCEClassifier(),

    # CanonicalIntervalForestClassifier(), #oxi
    # DrCIFClassifier(), #mallon oxi
    # SupervisedTimeSeriesForest(), #nai
    # TimeSeriesForestClassifier(), #nai

    # RocketClassifier(), #nai
    # Arsenal(), #nai
]

# Initialize lists to store classifier names, accuracies, and confusion matrices
classifier_names = []
accuracies = []
confusion_matrices = []
# Initialize lists to store results
execution_times = []
precisions = []
accuracies = []
auc_scores = []
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

    # Calculate precision, accuracy, AUC score, and F1 score
    precision = precision_score(y_test, predicted_labels, average='weighted')
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average='weighted')

    # Store the results
    precisions.append(precision)
    accuracies.append(accuracy)
    f1_scores.append(f1_score_val)

    # Calculate AUC score
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        if len(np.unique(y)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_score = auc(fpr, tpr)
        else:  # Multiclass classification
            auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            print(f"{classifier_name} AUC Score: {auc_score:.2f}")

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
