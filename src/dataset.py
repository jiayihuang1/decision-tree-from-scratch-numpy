import numpy as np
from numpy.random import default_rng

def load_dataset(filepath):
    """Load dataset from the specified filepath as NumPy array

    Args:
        filepath (str): The filepath to the dataset.txt file

    Returns:
        dataset (np.ndarray): NumPy array with shape (N, K), where N is the number of instances and K is the number of features + class label
    """

    # Load text dataset as NumPy array
    with open(filepath, "r") as file:
        dataset = np.loadtxt(file)

    return dataset


def train_test_k_fold_split(dataset, n_folds, random_generator=default_rng()):
    """Split dataset into k folds of train and test datasets.
    Use stratified splitting based on class label.

    Args:
        dataset (np.ndarray): NumPy array with shape (N, K), where N is the number of instances and K is the number of features + class label
        n_folds (int): Number of folds
        random_generator (np.random.Generator): A random generator

    Returns:
        k_folds (list): List of length n_folds. Each element in the list is a list
                    with two elements:
                    - Array of train instances
                    - Array of test instances
    """

    # Get unique class labels
    class_labels = np.unique(dataset[:, -1])

    # Initialize empty list to hold folds per class
    class_folds = []

    # Filter for each class label, shuffle and split into k folds
    for label in class_labels:
        label_dataset = dataset[dataset[:, -1] == label]
        shuffled_indices = random_generator.permutation(len(label_dataset))
        split_indices = np.array_split(shuffled_indices, n_folds)

        folds = []
        for k in range(n_folds):
            # Pick k as test
            test_indices = split_indices[k]
            train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

            # Create k fold datasets for this class label
            folds.append([label_dataset[train_indices], label_dataset[test_indices]])
        class_folds.append(folds)

    # Combine folds from all class labels
    k_folds = []
    for k in range(n_folds):
        train_dataset = np.zeros((0, dataset.shape[1]))
        test_dataset = np.zeros((0, dataset.shape[1]))

        for label_fold in class_folds:
            train_dataset = np.vstack((train_dataset, label_fold[k][0]))
            test_dataset = np.vstack((test_dataset, label_fold[k][1]))

        k_folds.append([train_dataset, test_dataset])

    return k_folds

