import numpy as np
import pandas as pd
import time
from sklearn.linear_model import RidgeClassifierCV
from rocket_functions import generate_kernels, apply_kernels
from sktime.datasets import load_UCR_UEA_dataset

results = []  # Initialize a list to store results


# Define a list of dataset names
dataset_names = ["Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECGFiveDays",
    "ElectricDevices",
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
# Loop over each dataset
for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")

    # Load the dataset
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    X_train = X_train.values
    X_test = X_test.values

    # Convert to float64
    X_train = np.array([x[0] for x in X_train]).astype(np.float64)
    X_test = np.array([x[0] for x in X_test]).astype(np.float64)

    # Start time measurement
    start_time = time.time()

    # Generate random kernels
    kernels = generate_kernels(X_train.shape[1], 10000)

    # Transform training set
    X_train_transform = apply_kernels(X_train, kernels)
    train_transform_time = time.time() - start_time

    # Train classifier
    start_time = time.time()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)
    training_time = time.time() - start_time

    # Transform test set and predict
    start_time = time.time()
    X_test_transform = apply_kernels(X_test, kernels)
    test_transform_time = time.time() - start_time

    start_time = time.time()
    predictions = classifier.predict(X_test_transform)
    test_time = time.time() - start_time

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)

    results.append({
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Training Transformation Time": train_transform_time,
        "Training Time": training_time,
        "Test Transformation Time": test_transform_time,  # Added test transformation time
        "Test Time": test_time,
    })

    # Modified print statements to match the second script
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Training Transformation Time: {train_transform_time}s")
    print(f"Training Time: {training_time}s")
    print(f"Test Transformation Time: {test_transform_time}s")  # Print test transformation time
    print(f"Test Time: {test_time}s")
    print("=" * 50)  # Separator for different datasets

# After processing all datasets, calculate the average accuracy and average time
average_accuracy = np.mean([result['Accuracy'] for result in results])
average_total_time = np.mean([
    result['Training Transformation Time'] +
    result['Training Time'] +
    result['Test Transformation Time'] +
    result['Test Time']
    for result in results
])

# Print the results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Total Time (Training Transformation + Training + Test Transformation + Test): {average_total_time}')
