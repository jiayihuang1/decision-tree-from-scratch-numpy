import numpy as np

def compute_confusion_matrix(y_true, y_prediction, class_labels=None):
    """Compute the confusion matrix

    Args:
        y_true (np.ndarray): Ground truth labels
        y_prediction (np.ndarray): Predicted labels
        class_labels (np.ndarray): Array of unique class labels. Defaults to the union of y_true and y_prediction

    Returns:
        np.array : Shape (C, C), where C is the number of classes.
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
        y_true (np.ndarray): Correct ground truth labels
        y_prediction (np.ndarray): Predicted labels

    Returns:
        float : Accuracy value
    """

    assert len(y_true) == len(y_prediction)

    try:
        return np.sum(y_true == y_prediction) / len(y_true)
    except ZeroDivisionError:
        return 0.
    

# def compute_precision(y_true, y_prediction):
#     """Compute the precision score per class given the ground truth and predictions

#     Also return the macro-averaged precision across classes.

#     Args:
#         y_true (np.ndarray): Correct ground truth labels
#         y_prediction (np.ndarray): Predicted labels

#     Returns:
#         precisions (np.ndarray): NumPy array of shape (C,), where each element is the precision for class c
#         macro_precision (float): Macro-averaged precision
#     """

#     confusion = compute_confusion_matrix(y_true, y_prediction)
#     precisions = np.zeros((len(confusion), ))
#     for c in range(confusion.shape[0]):
#         if np.sum(confusion[:, c]) > 0:
#             precisions[c] = confusion[c, c] / np.sum(confusion[:, c])

#     macro_precision = 0.
#     if len(precisions) > 0:
#         macro_precision = np.mean(precisions)

#     return precisions, macro_precision


# def compute_recall(y_true, y_prediction):
#     """Compute the recall score per class given the ground truth and predictions

#     Also return the macro-averaged recall across classes.

#     Args:
#         y_true (np.ndarray): Correct ground truth labels
#         y_prediction (np.ndarray): Predicted labels

#     Returns:
#         recalls (np.ndarray): NumPy array of shape (C,), where each element is the recall for class c
#         macro_r (float): Macro-averaged recall
#     """

#     confusion = compute_confusion_matrix(y_true, y_prediction)
#     recalls = np.zeros((len(confusion), ))
#     for c in range(confusion.shape[0]):
#         if np.sum(confusion[c, :]) > 0:
#             recalls[c] = confusion[c, c] / np.sum(confusion[c, :])

#     macro_recall = 0.
#     if len(recalls) > 0:
#         macro_recall = np.mean(recalls)

#     return recalls, macro_recall


# def compute_f1_score(y_true, y_prediction):
#     """Compute the F1-score per class given the ground truth and predictions

#     Also return the macro-averaged F1-score across classes.

#     Args:
#         y_true (np.ndarray): Correct ground truth labels
#         y_prediction (np.ndarray): Predicted labels

#     Returns:
#         f1s (np.ndarray): NumPy array of shape (C,), where each element is the F1-score for class c
#         macro_f1 (float): Macro-averaged F1-score
#     """

#     precisions, macro_precision = compute_precision(y_true, y_prediction)
#     recalls, macro_recall = compute_recall(y_true, y_prediction)

#     # Sanity check same length
#     assert len(precisions) == len(recalls)

#     f1s = np.zeros((len(precisions), ))
#     for c, (precision, recall) in enumerate(zip(precisions, recalls)):
#         if precision + recall > 0:
#             f1s[c] = 2 * precision * recall / (precision + recall)

#     # Compute the macro-averaged F1
#     macro_f1 = 0.
#     if len(f1s) > 0:
#         macro_f1 = np.mean(f1s)

#     return f1s, macro_f1

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
        test_db (np.ndarray): NumPy array of shape (N, D+1) where N is number of instances, D is number of features
        trained_tree (dict): Trained decision tree

    Returns:
        dict : Dictionary containing evaluation metrics
    """

    y_true = test_db[:, -1]
    y_prediction = predict(trained_tree, test_db[:, :-1])

    return compute_confusion_matrix(y_true, y_prediction)


def generate_classification_metrics(confusion_matrix):

    all_TP = []
    all_recall = []
    all_precision_rate = []
    all_F1 = []


    precisions = np.zeros((len(confusion_matrix), ))
    recalls = np.zeros((len(confusion_matrix), ))
    f1s = np.zeros((len(precisions), ))
    
    # generate metrics per class
    for c in range(len(confusion_matrix)):
        # TP = confusion_matrix[c][c]
        # FP = np.sum(confusion_matrix[:][c])
        # FN = np.sum(confusion_matrix[c][:])

        # precision_rate = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # F1 = (2 * precision_rate * recall) / (precision_rate + recall)

        if np.sum(confusion_matrix[c, :]) > 0:
            recalls[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[c, :])

        if np.sum(confusion_matrix[:, c]) > 0:
            precisions[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[:, c])

    for c, (precision, recall) in enumerate(zip(precisions, recalls)):
        if precision + recall > 0:
            f1s[c] = 2 * precision * recall / (precision + recall)

        # all_TP.append(TP)
        # all_precision_rate.append(precision_rate)
        # all_recall.append(recall)
        # all_F1.append(F1)

    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # accuracy = 100 * np.sum(all_TP) / np.sum(confusion_matrix)

    return accuracy, recalls, precisions, f1s
    # return accuracy, np.array(all_recall), np.array(all_precision_rate), np.array(all_F1)
