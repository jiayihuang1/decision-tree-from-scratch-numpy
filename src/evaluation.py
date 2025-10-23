import numpy as np


def compute_confusion_matrix(y_true, y_prediction, class_labels=None):
    """Compute the confusion matrix

    Args:
        y_true (np.ndarray): Ground truth labels
        y_prediction (np.ndarray): Predicted labels
        class_labels (np.ndarray): Array of unique class labels. Defaults to the union of y_true and y_prediction

    Returns:
        confusion (np.ndarray) : Shape (C, C), where C is the number of classes.
                            Rows are ground truth per class, columns are predictions
    """

    # If no class_labels given, obtain set of unique class labels from union of ground truth and prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_true, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # For each correct class (row), compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # Filter for predictions where corresponding ground truth == enumerated class label
        indices = (y_true == label)
        predictions = y_prediction[indices]

        # Compute counts per label
        unique_labels, counts = np.unique(predictions, return_counts=True)

        # Convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # Fill up the confusion matrix
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def compute_accuracy(y_true, y_prediction):
    """Compute the accuracy given the ground truth and predictions

    Args:
        y_true (np.ndarray): Ground truth labels
        y_prediction (np.ndarray): Predicted labels

    Returns:
        float : Accuracy value
    """

    assert len(y_true) == len(y_prediction)

    try:
        return np.sum(y_true == y_prediction) / len(y_true)
    except ZeroDivisionError:
        return 0.


def predict(decision_tree, x_test):
    """Predict class label for given instance using decision tree

    Args:
        decision_tree (dict): Decision tree nodes in nested dictionary format
        x_test (np.ndarray): NumPy array with shape (N, K-1), where K-1 is the number of features

    Returns:
        y_prediction (np.ndarray): Array of predicted class labels
    """

    y_prediction = np.zeros((x_test.shape[0],), dtype=int)

    for i in range(x_test.shape[0]):
        # Traverse tree until leaf node is reached
        current_node = decision_tree

        while current_node["leaf"] != True:
            attribute = current_node["attribute"]
            value = current_node["value"]
            if x_test[i, attribute] <= value:
                current_node = current_node["left"]
            elif x_test[i, attribute] > value:
                current_node = current_node["right"]

        # Return majority class label at leaf node
        y_prediction[i] = current_node["prediction"]

    return y_prediction


def evaluate(test_db, trained_tree):
    """Evaluate the trained decision tree on the test dataset

    Args:
        test_db (np.ndarray): NumPy array of shape (N, K) where N is number of instances, K is number of features + class labels
        trained_tree (dict): Trained decision tree

    Returns:
        np.ndarray : Shape (C, C), where C is the number of classes.
                    Rows are ground truth per class, columns are predictions
    """

    y_true = test_db[:, -1]
    y_prediction = predict(trained_tree, test_db[:, :-1])

    return compute_confusion_matrix(y_true, y_prediction)


def generate_classification_metrics(confusion_matrix):
    """Compute classification metrics from given confusion matrix

    Args:
        confusion_matrix (np.ndarray): Confusion matrix of shape (C, C)

    Returns:
        (float, np.ndarray, np.ndarray, np.ndarray): Tuple containing accuracy, recalls, precisions, f1s
    """

    # Initialize lists to hold per class metrics
    all_TP = []
    all_recall = []
    all_precision_rate = []
    all_f1 = []

    # Compute metrics per class
    for c in range(len(confusion_matrix)):
        TP = confusion_matrix[c, c]
        FP = np.sum(confusion_matrix[:, c]) - TP
        FN = np.sum(confusion_matrix[c, :]) - TP

        precision_rate = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision_rate * recall) / (precision_rate + recall)

        all_TP.append(TP)
        all_precision_rate.append(precision_rate)
        all_recall.append(recall)
        all_f1.append(f1)

    accuracy = 100 * np.sum(all_TP) / np.sum(confusion_matrix)

    return accuracy, np.array(all_recall), np.array(all_precision_rate), np.array(all_f1)
