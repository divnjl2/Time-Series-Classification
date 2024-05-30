import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from ConvFS_functions import generate_kernels, transform_and_select_features
from sktime.datasets import load_UCR_UEA_dataset
import time
from memory_profiler import memory_usage


# Define a list of dataset names
dataset_names = [
    "ArrowHead",
    "FiftyWords",
    "WordSynonyms",
    "Car",
    "CricketX",
    "ShapesAll",
    "Rock",
    "ACSF1",
    "ERing",
    "Handwriting",
    "HandMovementDirection"

]

""""FaceAll",
    "FacesUCR",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Plane",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Wine",
    "Worms",
    "WormsTwoClass",
    "Yoga" """
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

total_start_time = time.time()
results = []
for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    # Convert DataFrame to numpy array if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = np.stack(X_train.iloc[:, 0].apply(lambda x: x.to_numpy() if isinstance(x, pd.Series) else x))
    if isinstance(X_test, pd.DataFrame):
        X_test = np.stack(X_test.iloc[:, 0].apply(lambda x: x.to_numpy() if isinstance(x, pd.Series) else x))

    avg_series_length = np.mean([len(x) for x in X_train])

    # Generate kernels based on the average series length
    kernels = generate_kernels(X_train.shape[1], 10000, int(avg_series_length))

    # Transform and select features for training data
    X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(X_train, kernels, y_train, is_train=True)

    # Measure memory usage during model training
    train_mem_usage = memory_usage((lambda: classifier.fit(X_train_transformed, y_train)), max_usage=True)

    # Transform test data
    X_test_transformed = transform_and_select_features(X_test, kernels, selector=selector, scaler=scaler, is_train=False)

    # Measure memory usage during making predictions
    prediction_mem_usage = memory_usage((lambda: classifier.predict(X_test_transformed)), max_usage=True)

    # Test classifier
    predictions = classifier.predict(X_test_transformed)
    accuracy = np.mean(predictions == y_test)

    results.append({
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Num Features": best_num_features,
        "Training Memory Usage (MB)": train_mem_usage,
        "Prediction Memory Usage (MB)": prediction_mem_usage,
    })

    # Print the results
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Number of Features: {best_num_features}")
    print(f"Training Memory Usage (MB): {train_mem_usage}")
    print(f"Prediction Memory Usage (MB): {prediction_mem_usage}")
    print("=" * 50)  # Separator for different datasets

# After processing all datasets, calculate the average accuracy
average_accuracy = np.mean([result['Accuracy'] for result in results])

# Print the results
print(f'Average Accuracy: {average_accuracy}')

total_time = time.time() - total_start_time
print(f"Total processing time: {total_time} seconds")