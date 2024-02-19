import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from ConvFS_functions import generate_kernels, transform_and_select_features
from sktime.datasets import load_UCR_UEA_dataset
import time
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import roc_curve, auc
from memory_profiler import memory_usage
from sklearn.preprocessing import label_binarize

# Define a list of dataset names
dataset_names = ["HandMovementDirection"]

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

    # Measure memory usage for training transformation
    start_mem = memory_usage()[0]
    start_time = time.time()
    kernels = generate_kernels(X_train.shape[1], 10000, int(avg_series_length))
    X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(X_train, kernels, y_train, is_train=True)
    train_transform_time = time.time() - start_time
    train_transform_mem = memory_usage()[0] - start_mem

    # Train classifier
    start_time = time.time()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transformed, y_train)
    training_time = time.time() - start_time
    training_mem = memory_usage()[0] - start_mem

    # Measure memory usage for test transformation
    start_time = time.time()
    X_test_transformed = transform_and_select_features(X_test, kernels, selector=selector, scaler=scaler, is_train=False)
    test_transform_time = time.time() - start_time
    test_transform_mem = memory_usage()[0] - start_mem

    # Test classifier
    start_time = time.time()
    predictions = classifier.predict(X_test_transformed)
    test_time = time.time() - start_time
    test_mem = memory_usage()[0] - start_mem

    accuracy = np.mean(predictions == y_test)
    precision = precision_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    results.append({
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1-Score": f1,
        "Num Features": best_num_features,
        "Training Transformation Time": train_transform_time,
        "Training Time": training_time,
        "Test Transformation Time": test_transform_time,
        "Test Time": test_time,
        "Training Transformation Memory Usage": train_transform_mem,
        "Training Memory Usage": training_mem,
        "Test Transformation Memory Usage": test_transform_mem,
        "Test Memory Usage": test_mem
    })

    # Print the results for ConvFS
    for key, value in results[-1].items():
        print(f"{key}: {value}")

total_time = time.time() - total_start_time
print(f'Total execution time: {total_time}')


    # Print the results
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"F1-Score: {f1}")
    print(f"Number of Features: {best_num_features}")  # Print number of features used
    print(f"Training Transformation Time: {train_transform_time}s")
    print(f"Training Time: {training_time}s")
    print(f"Test Transformation Time: {test_transform_time}s")
    print(f"Test Time: {test_time}s")
    #print(f"memory usage: {max_mem_usage}")
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

"""# Print the results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Total Time (Training Transformation + Training + Test Transformation + Test): {average_total_time}')

total_time = time.time() - total_start_time
print(total_time)"""




from sklearn.calibration import CalibratedClassifierCV

# First, fit the RidgeClassifierCV
ridge_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
ridge_clf.fit(X_train_transformed, y_train)

# Then, use the fitted classifier with CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(estimator=ridge_clf, method='sigmoid', cv='prefit')  # Note: Use `estimator`, not `base_estimator` if warning suggests
calibrated_clf.fit(X_train_transformed, y_train)

# Now you can use predict_proba with the calibrated classifier
y_proba = calibrated_clf.predict_proba(X_test_transformed)

# Proceed with ROC AUC calculation as before


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have already defined y_test and y_train
classes = np.unique(np.concatenate((y_train, y_test)))
y_test_binarized = label_binarize(y_test, classes=classes)

if hasattr(calibrated_clf, "predict_proba"):
    y_proba = calibrated_clf.predict_proba(X_test_transformed)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_binarized.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-average ROC AUC: {roc_auc['macro']:.3f}")
    print(f"Macro-average FPR: {fpr['macro']}")
    print(f"Macro-average TPR: {tpr['macro']}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["macro"], tpr["macro"], label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} Macro-average ROC curve for ConvFS')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("Classifier does not support probability predictions after calibration.")


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming predictions and y_test are defined from your ConvFS evaluation
conf_matrix = confusion_matrix(y_test, predictions)

# Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for ConvFS')
plt.show()

# Plotting ROC AUC for each class
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC AUC for ConvFS')
plt.legend(loc='lower right')
plt.show()
