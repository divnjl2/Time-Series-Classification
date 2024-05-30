import numpy as np
from sklearn.metrics import accuracy_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sktime.datasets import load_UCR_UEA_dataset
import time
from collections import Counter
from memory_profiler import memory_usage

# Distance-based:
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier


dataset_name = [
    "SemgHandGenderCh2"]
for dataset in dataset_name:
    # Load the dataset
    X_train_raw, y_train = load_UCR_UEA_dataset(dataset, split="train", return_X_y=True)  # Use `dataset` here
    X_test_raw, y_test = load_UCR_UEA_dataset(dataset, split="test", return_X_y=True)  # And here as well

    # Print dataset sizes and class distribution
    print("Length of each time series:", X_train_raw.iloc[0, 0].size)
    print("Train size:", len(y_train))
    print("Test size:", len(y_test))
    print("Training set class distribution:", Counter(y_train))
    print("Test set class distribution:", Counter(y_test))


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
    classifiers = [KNeighborsTimeSeriesClassifier()]

    # Initialize lists to store results
    results = {"Execution Time": [],"Accuracy": []}


    # Function to evaluate classifier
    def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
        # Inner function to include both fitting and prediction for memory profiling
        def fit_and_predict():
            classifier.fit(X_train, y_train)
            return classifier.predict(X_test)

        # Measure execution time and memory usage for fitting and predicting
        start_time = time.time()
        mem_usage = memory_usage((fit_and_predict,), interval=0.1, include_children=True, retval=True)
        execution_time = time.time() - start_time
        max_mem_usage = max(mem_usage[0]) - min(mem_usage[0])  # mem_usage[0] contains the memory usage
        predicted_labels = mem_usage[1]  # mem_usage[1] contains the return value from fit_and_predict

        # Proceed with the rest of the evaluation
        accuracy = accuracy_score(y_test, predicted_labels)


        # If the classifier supports probability estimates, calculate ROC AUC scores
        roc_auc_macro = roc_auc_micro = None
        if hasattr(classifier, "predict_proba"):
            y_prob = classifier.predict_proba(X_test)


        # Return all the metrics including memory usage
        return execution_time, accuracy



    # Evaluate each classifier
    for classifier in classifiers:
        classifier_name = type(classifier).__name__
        # Use the resampled data if resampling was done, else use the original data

        exec_time, accuracy = \
            evaluate_classifier(classifier, X_train_flat, X_test_flat, y_train, y_test)


        results["Execution Time"].append(exec_time)
        results["Accuracy"].append(accuracy)

        # Print results
        print(dataset)
        print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
        print(f"{classifier_name} Accuracy: {accuracy:.2f}")