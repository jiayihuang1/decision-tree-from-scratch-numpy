import numpy as np
import copy

from src.evaluation import compute_accuracy, predict


def compute_entropy(dataset):
    """Compute entropy of the given dataset

    Args:
        dataset (np.ndarray): The input dataset

    Returns:
        entropy (float): The computed entropy
    """

    unique_labels, counts = np.unique(dataset[:, -1], return_counts=True)
    prob_distribution = counts / len(dataset[:, -1])
    entropy = -np.sum(prob_distribution * np.log2(prob_distribution))

    return entropy


def compute_information_gain(dataset, attribute, value):
    """Compute information gain of a split on the given dataset

    Args:
        dataset (np.ndarray): The input dataset
        attribute (int): The index of the attribute to split on
        value (float): The value to split the attribute on

    Returns:
        information_gain (float): The computed information gain
    """

    # Split the dataset by the attribute and value
    left_dataset, right_dataset = dataset[dataset[:, attribute] <= value], dataset[dataset[:, attribute] > value]

    # Compute entropies and information gain
    left_entropy, right_entropy = compute_entropy(left_dataset), compute_entropy(right_dataset)
    overall_entropy = compute_entropy(dataset)
    remainder = (len(left_dataset) / len(dataset)) * left_entropy + (len(right_dataset) / len(dataset)) * right_entropy
    information_gain = overall_entropy - remainder

    return information_gain


def decision_tree_learning(dataset, current_depth=0):
    """Recursively train a decision tree given an input dataset

    The recursive function takes in the subset of data and current depth of the node and returns the 'filled'
    node to the decision tree that is being trained and the function is called again for the children nodes
    (if any) until a leaf node is reached.

    Args:
        dataset (np.ndarray): The input dataset
        current_depth (int): The current depth of the tree

    Returns:
        node (dict): The trained decision tree
        int : The max depth of the trained decision tree
    """

    # Leaf node: all data points are pure (i.e., same class label)
    if compute_entropy(dataset) == 0:
        leaf_node = {
            'prediction': dataset[0][-1],
            'depth': current_depth,
            'leaf': True
        }
        return leaf_node, current_depth

    else:
        # Edge case: Information gain is zero, no further splits possible
        optimal_attribute, optimal_value = find_split(dataset)
        if optimal_attribute is None or optimal_value is None:

            leaf_node = {
                'prediction': majority_class(dataset),
                'depth': current_depth,
                'leaf': True
            }
            return leaf_node, current_depth

        else:
            # Recursively split the dataset and build subtrees
            left_dataset = dataset[dataset[:, optimal_attribute] <= optimal_value]
            right_dataset = dataset[dataset[:, optimal_attribute] > optimal_value]
            next_depth = current_depth + 1

            left_subtree, left_max_depth = decision_tree_learning(left_dataset, next_depth)
            right_subtree, right_max_depth = decision_tree_learning(right_dataset, next_depth)

            node = {
                'attribute': optimal_attribute,
                'value': optimal_value,
                'left': left_subtree,
                'right': right_subtree,
                'depth': current_depth,
                'leaf': False
            }

            return node, max(left_max_depth, right_max_depth)


def find_split(dataset):
    """Compute optimal attribute to split dataset on based on max information gain

    Args:
        dataset (np.ndarray): The input dataset

    Returns:
        optimal_attribute (int): The index of the optimal attribute to split on
        optimal_value (float): The optimal value to split the optimal attribute on
    """

    highest_IG = 0
    optimal_attribute = None
    optimal_value = None

    for attribute in range(len(dataset[0]) - 1):
        # Compute median value of attribute to split on
        attribute_value = np.median(dataset[:, attribute])
        information_gain = compute_information_gain(dataset, attribute, attribute_value)

        if information_gain > highest_IG:
            highest_IG = information_gain
            optimal_attribute = attribute
            optimal_value = attribute_value

    return optimal_attribute, optimal_value


def majority_class(current_dataset):
    """Compute majority class label of the given dataset

    Args:
        current_dataset (np.ndarray): The input dataset

    Returns:
        int : The majority class label
    """

    unique_labels, counts = np.unique(current_dataset[:, -1], return_counts=True)
    max_count_index = np.argmax(counts)
    return unique_labels[max_count_index]


def prune_n_parses(root_node, current_data_subset, validation_data):
    """Fires recursive 'prune' function and checks if further pruning is required.
    Reverts to previous tree state if pruning decreases validation accuracy.

    Args:
        root_node (dict): Dictionary of the decision tree
        current_data_subset (np.ndarray): The input dataset
        validation_data (np.ndarray): The validation dataset

    Returns:
        pruned_tree (dict): The pruned decision tree
    """

    pruned_tree = root_node

    prune_flag = True  # To control the pruning loop
    n_parses = 0  # Total number of times the 'prune' function has been called

    # Calculate the starting accuracy on the VALIDATION SET
    y_true = validation_data[:, -1]
    y_prediction_before = predict(pruned_tree, validation_data)
    best_accuracy = compute_accuracy(y_true, y_prediction_before)

    while prune_flag:

        n_parses += 1

        # Store the tree before the current pruning pass
        tree_before_prune = copy.deepcopy(pruned_tree)

        # Perform one full pass of pruning over the tree
        pruned_tree = prune(pruned_tree, pruned_tree, current_data_subset, validation_data)

        # Calculate accuracy on the VALIDATION SET AFTER pruning pass
        y_prediction_after = predict(pruned_tree, validation_data)
        accuracy_after = compute_accuracy(y_true, y_prediction_after)

        if accuracy_after > best_accuracy:
            # Pruning improved performance
            best_accuracy = accuracy_after

        else:
            # Pruning did not improve performance
            prune_flag = False

            # If accuracy dropped, revert to the tree state before this pass
            if accuracy_after < best_accuracy:
                pruned_tree = tree_before_prune  # Revert to the last best tree
                break

    return pruned_tree


def prune(root_node, current_node, current_data_subset, validation_data):
    """Recursive function to prune the decision tree

    Args:
        root_node (dict): Dictionary of the decision tree from the root
        current_node (dict): Dictionary of the current node being evaluated
        current_data_subset (np.ndarray): The input dataset subset for the current node
        validation_data (np.ndarray): The validation dataset

    Returns:
        root_node (dict): The pruned decision tree
    """

    # Stop recursion if the current_node is a leaf
    if current_node["leaf"]:
        return

    # Check if this node is connected to two leaves
    if current_node["left"]["leaf"] and current_node["right"]["leaf"]:
        # Store current children nodes
        ori_left_node = current_node["left"]
        ori_right_node = current_node["right"]

        # Calculate error BEFORE pruning
        y_true = validation_data[:, -1]
        y_prediction_before = predict(root_node, validation_data)
        accuracy_before = compute_accuracy(y_true, y_prediction_before)

        # Temporarily collapse the split (simulate pruning)
        current_node["left"] = None
        current_node["right"] = None
        current_node["leaf"] = True
        current_node["prediction"] = majority_class(current_data_subset)

        # Calculate error AFTER pruning
        y_prediction_after = predict(root_node, validation_data)
        accuracy_after = compute_accuracy(y_true, y_prediction_after)

        # Decision: Does pruning improve the validation accuracy?
        if accuracy_after >= accuracy_before:
            pass  # Make pruning permanent
        else:
            # Revert pruning
            current_node["left"] = ori_left_node
            current_node["right"] = ori_right_node
            current_node["leaf"] = False
            del current_node["prediction"]

    else:
        left_node = current_node["left"]
        right_node = current_node["right"]

        # Update training data subset to each children's data subset
        current_attribute, current_value = current_node["attribute"], current_node["value"]
        left_data_subset = current_data_subset[current_data_subset[:, current_attribute] <= current_value]
        right_data_subset = current_data_subset[current_data_subset[:, current_attribute] > current_value]

        # Recursively calls the prune function for children nodes
        prune(root_node, left_node, left_data_subset, validation_data)
        prune(root_node, right_node, right_data_subset, validation_data)

    return root_node

