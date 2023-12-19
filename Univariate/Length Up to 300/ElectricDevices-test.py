import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from sktime.datasets import load_UCR_UEA_dataset

# IMPORT CLASSIFIERS:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Deep Learning:
from sklearn.neural_network import MLPClassifier # We can also import it through sktime: from sktime.classification.deep_learning.mlp import MLPClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning.fcn import FCNClassifier
from sktime.classification.deep_learning.macnn import MACNNClassifier
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier

# Dictionary-based:
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.classification.dictionary_based import IndividualTDE
from sktime.classification.dictionary_based import MUSE
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.dictionary_based import WEASEL

# Distance-based:
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based import ProximityForest
from sktime.classification.distance_based import ProximityStump
from sktime.classification.distance_based import ProximityTree

# Feature-based:
from sktime.classification.feature_based import MatrixProfileClassifier

# Interval-based:
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.interval_based import DrCIF
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.interval_based import TimeSeriesForestClassifier

# Kernel-based:
from sktime.classification.kernel_based import Arsenal
from sktime.classification.kernel_based import RocketClassifier

#check shapelet transform


# Deep learning
#from aeon.classification.deep_learning.mlp import MLPClassifier as AeonMLPClassifier ###############
#from aeon.classification.deep_learning.cnn import CNNClassifier ###############
#from aeon.classification.deep_learning.fcn import FCNClassifier ###############
#from aeon.classification.deep_learning import InceptionTimeClassifier
#from aeon.classification.deep_learning import IndividualInceptionClassifier
#from aeon.classification.deep_learning.tapnet import TapNetClassifier ###############

# Dictionary-based
#from aeon.classification.dictionary_based import BOSSEnsemble ###############
#from aeon.classification.dictionary_based import ContractableBOSS ###############
#from aeon.classification.dictionary_based import IndividualBOSS ###############
#from aeon.classification.dictionary_based import MUSE ###############
#from aeon.classification.dictionary_based import TemporalDictionaryEnsemble ###############
#from aeon.classification.dictionary_based import WEASEL ###############
#from aeon.classification.dictionary_based import WEASEL_V2

#Distance-based
#from aeon.classification.distance_based import ShapeDTW
#from aeon.classification.distance_based import ElasticEnsemble ###############
#from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier as AeonKNeighborsTimeSeriesClassifier ###############

#Feature-based
#from aeon.classification.feature_based import Catch22Classifier
#from aeon.classification.feature_based import MatrixProfileClassifier ###############

#Interval-based
#from aeon.classification.interval_based import CanonicalIntervalForestClassifier ###############
#from aeon.classification.interval_based import DrCIFClassifier ###############
#from aeon.classification.interval_based import SupervisedTimeSeriesForest ###############
#from aeon.classification.interval_based import TimeSeriesForestClassifier ###############
#from aeon.classification.interval_based import RandomIntervalClassifier

#Shapelet-based
#from aeon.classification.shapelet_based import ShapeletTransformClassifier
#from aeon.classification.shapelet_based import RDSTClassifier

# sklearn
#from aeon.classification.sklearn import ContinuousIntervalTree
#from aeon.classification.sklearn import RotationForestClassifier

# Convolution-based
#from aeon.classification.convolution_based import Arsenal ###############
#from aeon.classification.convolution_based import RocketClassifier ###############

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
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    MLPClassifier(),
    CNNClassifier(n_epochs=100),
   # FCNClassifier(),
    #MACNNClassifier(),
   # MCDCNNClassifier(),
   # ResNetClassifier(),
   # TapNetClassifier(),
    BOSSEnsemble(),
    ContractableBOSS(),
    IndividualBOSS(),
    IndividualTDE(),
    MUSE(),
    TemporalDictionaryEnsemble(),
    WEASEL(),
    ElasticEnsemble(),
    KNeighborsTimeSeriesClassifier(),
    ProximityForest(),
    ProximityStump(),
    ProximityTree(),
    MatrixProfileClassifier(),
    CanonicalIntervalForest(),
    DrCIF(),
    SupervisedTimeSeriesForest(),
    TimeSeriesForestClassifier(),
   # Arsenal(),
    #RocketClassifier()
]


"""# In case you want to run more classifiers:
# Define a list of classifiers
classifiers = [
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    MLPClassifier(),
    #KNeighborsTimeSeriesClassifier(n_neighbors=1),  # K-Nearest Neighbors for time series
    Arsenal(),
    RocketClassifier(),
    CNNClassifier(),
    FCNClassifier(),
    AeonMLPClassifier(),
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
    AeonKNeighborsTimeSeriesClassifier(),
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
"""

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
plt.figure(figsize=(20, 15))
plt.bar(classifier_names, accuracies, color='skyblue')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')  # Set rotation to 45 degrees and adjust alignment
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
