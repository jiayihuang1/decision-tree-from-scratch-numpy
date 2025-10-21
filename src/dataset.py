import numpy as np
from numpy.random import default_rng

def load_dataset(filepath):
    """Load dataset from the specified filepath as NumPy array

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        dataset (np.ndarray): NumPy array with shape (N, K), where N is the number of instances and K is the number of features + class label
    """

    # Load text dataset as NumPy array
    with open(filepath, "r") as file:
        dataset = np.loadtxt(file)

    return dataset


# def train_test_split(dataset, test_proportion=0.2, random_generator=default_rng()):
#     """Split dataset into train and test datasets, according to test_proportion. Use stratified splitting based on class label

#     Args:
#         dataset (np.ndarray): NumPy array with shape (N, K), where N is the number of instances and K is the number of features + class label
#         test_proportion (float): Proportion of dataset (0.0 - 1.0) to split as test dataset
#         random_generator (np.random.Generator): A random generator

#     Returns:
#         train_dataset (np.ndarray): Train instances shape (N_train, K)
#         test_dataset (np.ndarray): Test instances shape (N_test, K)
#     """

#     # Get unique class labels
#     class_labels = np.unique(dataset[:, -1])

#     # Initialize empty train and test dataset arrays
#     train_dataset = np.zeros((0, dataset.shape[1]))
#     test_dataset = np.zeros((0,dataset.shape[1]))

#     # Filter for each class label, shuffle and split into train and test according to test_proportion
#     for label in class_labels:
#         label_dataset = dataset[dataset[:, -1] == label]
#         shuffled_indices = random_generator.permutation(len(label_dataset))
#         n_test = round(len(label_dataset) * test_proportion)
#         n_train = len(label_dataset) - n_test

#         label_train = label_dataset[shuffled_indices[:n_train]]
#         train_dataset = np.vstack((label_train, train_dataset))

#         label_test = label_dataset[shuffled_indices[:n_train]]
#         test_dataset = np.vstack((label_test, test_dataset))

#     return train_dataset, test_dataset


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


# def train_val_test_k_fold_split(dataset, n_folds, random_generator=default_rng()):
#     """Split dataset into k folds of train, validation and test datasets.
#     Use stratified splitting based on class label.

#     Args:
#         dataset (np.ndarray): NumPy array with shape (N, K), where N is the number of instances and K is the number of features + class label
#         n_folds (int): Number of folds
#         random_generator (np.random.Generator): A random generator

#     Returns:
#         k_folds (list): List of length n_folds. Each element in the list is a list
#                     with three elements:
#                     - Array of train instances
#                     - Array of validation instances
#                     - Array of test instances
#     """

#     # Get unique class labels
#     class_labels = np.unique(dataset[:, -1])

#     # Initialize empty list to hold folds per class
#     class_folds = []

#     # Filter for each class label, shuffle and split into k folds
#     for label in class_labels:
#         label_dataset = dataset[dataset[:, -1] == label]
#         shuffled_indices = random_generator.permutation(len(label_dataset))
#         split_indices = np.array_split(shuffled_indices, n_folds)

#         folds = []
#         for k in range(n_folds):
#             # Pick k as test, and k+1 as validation (or 0 if k is the final split)
#             test_indices = split_indices[k]
#             val_indices = split_indices[(k+1) % n_folds]

#             # Concatenate remaining splits for train
#             train_indices = np.zeros((0, ), dtype=int)
#             for i in range(n_folds):
#                 # Concatenate to train set if not validation or test
#                 if i not in [k, (k+1) % n_folds]:
#                     train_indices = np.hstack([train_indices, split_indices[i]])

#             # Create k fold datasets for this class label
#             folds.append([label_dataset[train_indices], label_dataset[val_indices], label_dataset[test_indices]])
#         class_folds.append(folds)

#     # Combine folds from all class labels
#     k_folds = []
#     for k in range(n_folds):
#         train_dataset = np.zeros((0, dataset.shape[1]))
#         val_dataset = np.zeros((0, dataset.shape[1]))
#         test_dataset = np.zeros((0, dataset.shape[1]))

#         for label_fold in class_folds:
#             train_dataset = np.vstack((train_dataset, label_fold[k][0]))
#             val_dataset = np.vstack((val_dataset, label_fold[k][1]))
#             test_dataset = np.vstack((test_dataset, label_fold[k][2]))

#         k_folds.append([train_dataset, val_dataset, test_dataset])

#     return k_folds


# # if __name__ == "__main__":
#     # test_load_dataset()
#     # test_train_test_split()