import numpy as np

from src.evaluation import compute_accuracy, predict
import copy

def compute_entropy(dataset):
    """ function to compute entropy """

    unique_labels, counts = np.unique(dataset[:, -1], return_counts=True)
    prob_distribution = counts / len(dataset[:, -1])
    entropy = -np.sum(prob_distribution * np.log2(prob_distribution))

    return entropy


def compute_information_gain(dataset, attribute, value):
    """ function to compute information gain """

    left_dataset, right_dataset = dataset[dataset[:, attribute] <= value], dataset[dataset[:, attribute] > value]
    left_entropy, right_entropy = compute_entropy(left_dataset), compute_entropy(right_dataset)
    overall_entropy = compute_entropy(dataset)
    remainder = (len(left_dataset) / len(dataset)) * left_entropy + (len(right_dataset) / len(dataset)) * right_entropy
    information_gain = overall_entropy - remainder

    return information_gain


def decision_tree_learning(dataset, current_depth=0):
    """ Recursive function to build the decision tree """
    if compute_entropy(dataset) == 0:
        leaf_node = {
            'prediction': dataset[0][-1],
            'depth': current_depth,
            'leaf': True
        }
        return leaf_node, current_depth

    else:
        # Edge case: Info gain is zero
        optimal_attribute, optimal_value = find_split(dataset)
        if optimal_attribute is None or optimal_value is None:
            
            leaf_node = {
                'prediction': majority_class(dataset),
                'depth': current_depth,
                'leaf': True
            }
            return leaf_node, current_depth

        else:
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
    # last column is y label
    highest_IG = 0
    optimal_attribute = None
    optimal_value = None

    for attribute in range(len(dataset[0]) - 1):
        # median value of feature
        attribute_value = np.median(dataset[:, attribute])

        information_gain = compute_information_gain(dataset, attribute, attribute_value)

        if information_gain > highest_IG:
            highest_IG = information_gain
            optimal_attribute = attribute
            optimal_value = attribute_value

    return optimal_attribute, optimal_value


def majority_class(current_dataset):
    unique_labels, counts = np.unique(current_dataset[:, -1], return_counts=True)
    max_count_index = np.argmax(counts)
    return unique_labels[max_count_index]


def prune_n_parses(root_node, current_data_subset, validation_data):

    pruned_tree = root_node

    prune_flag = True  # number of consecutive parses that did NOT improve accuracy
    n_parses = 0  # total number of times the 'prune' function has been called

    # Calculate the starting accuracy on the VALIDATION SET
    y_true = validation_data[:, -1]
    y_prediction_before = predict(pruned_tree, validation_data)
    best_accuracy = compute_accuracy(y_true, y_prediction_before)

    while prune_flag:

        n_parses += 1

        # store the tree before the current pruning pass
        tree_before_prune = copy.deepcopy(pruned_tree)

        # perform one full pass of pruning over the tree
        pruned_tree = prune(pruned_tree, pruned_tree, current_data_subset, validation_data)

        # calculate accuracy on the VALIDATION SET AFTER pruning pass
        y_prediction_after = predict(pruned_tree, validation_data)
        accuracy_after = compute_accuracy(y_true, y_prediction_after)

        if accuracy_after > best_accuracy:
            # pruning improved performance
            best_accuracy = accuracy_after

        else:
            # pruning did not improve performance
            prune_flag = False

            # if accuracy dropped, revert to the tree state before this pass
            if accuracy_after < best_accuracy:
                pruned_tree = tree_before_prune  # Revert to the last best tree
                print(f"Pruning stopped: No improvement for {n_parses} parses.\n")
                break

        print(f"Number of parses: {n_parses}. Validation Accuracy: {accuracy_after:.4f}")

    return pruned_tree


def prune(root_node, current_node, current_data_subset, validation_data):
    """ function to prune the tree based on lowering validation error """

    # stop recursion if the current_node is a leaf
    if current_node["leaf"]:
        return

    # check if this node is connected to two leaves
    if current_node["left"]["leaf"] and current_node["right"]["leaf"]:
        # store current children nodes
        ori_left_node = current_node["left"]
        ori_right_node = current_node["right"]

        # calculate error BEFORE pruning
        y_true = validation_data[:, -1]
        y_prediction_before = predict(root_node, validation_data)
        accuracy_before = compute_accuracy(y_true, y_prediction_before)

        # temporarily collapse the split (simulate pruning)
        current_node["left"] = None
        current_node["right"] = None
        current_node["leaf"] = True
        current_node["prediction"] = majority_class(current_data_subset)

        # calculate error AFTER pruning
        y_prediction_after = predict(root_node, validation_data)
        accuracy_after = compute_accuracy(y_true, y_prediction_after)

        # decision: does the pruning improve the validation accuracy?
        if accuracy_after >= accuracy_before:
            pass # make pruning permanent
        else:
            # revert pruning
            current_node["left"] = ori_left_node
            current_node["right"] = ori_right_node
            current_node["leaf"] = False
            del current_node["prediction"]


    else:
        left_node = current_node["left"]
        right_node = current_node["right"]

        # updating training data subset to each children's data subset
        current_attribute, current_value = current_node["attribute"], current_node["value"]
        left_data_subset = current_data_subset[current_data_subset[:, current_attribute] <= current_value]
        right_data_subset = current_data_subset[current_data_subset[:, current_attribute] > current_value]

        # recursively calls the prune function for children nodes
        prune(root_node, left_node, left_data_subset, validation_data)
        prune(root_node, right_node, right_data_subset, validation_data)

    return root_node







# # TODO: Reference

# def compute_split_point(x):
#     """Determine split point of feature array. Take midpoint of sorted array of feature instances.

#     Args:
#         x (np.ndarray): NumPy array with shape (N, K-1), where N is the number of instances and K-1 is the number of features
    
#     Returns:
#         split_points (float): Array of split points for each feature, shape (1, K-1)
#     """

#     # Sort and return median or average of two middle values (for even sized arrays)
#     split_points = np.median(x, axis=0)
#     split_points = split_points.reshape((1, -1))

#     return split_points


# def split_dataset(x, y, split_points, entropy_total):
#     """Split dataset into left and right subsets based on max entropy

#     Args:
#         x (np.ndarray): NumPy array with shape (N, K-1), where N is the number of instances and K-1 is the number of features
#         y (np.ndarray): NumPy array with shape (N, ) with integers from 1 to C, where C is the number of classes
#         split_points (np.ndarray): Array of split points for each feature, shape (1, K-1)
#         entropy_total (float): Entropy of initial unsplit dataset

#     Returns:
#         final_x_left (np.ndarray): Left split feature array with shape (N_left, K-1)
#         final_y_left (np.ndarray): Left split label array with shape (N_left, 1)
#         final_x_right (np.ndarray): Right split feature array with shape (N_right, K-1)
#         final_y_right (np.ndarray): Right split label array with shape (N_right, 1)
#         final_feature_index (int): Index of feature used for split
#         final_split_point (float): Split point value used for split
#     """
    
#     n_features = split_points.shape[1]

#     # Initialize variables to track best split
#     final_x_left = None
#     final_y_left = None
#     final_x_right = None
#     final_y_right = None
#     final_feature_index = None
#     final_split_point = None
#     max_information_gain = 0.

#     # Iterate over each feature and split by computed split point
#     for feature_index in range(n_features):
#         split_point = split_points[0, feature_index]

#         left_indices = (x[:, feature_index] <= split_point)
#         right_indices = (x[:, feature_index] > split_point)

#         x_left = x[left_indices]
#         y_left = y[left_indices]
#         entropy_left = compute_entropy(y_left)

#         x_right = x[right_indices]
#         y_right = y[right_indices]
#         entropy_right = compute_entropy(y_right)

#         # Track split with maximum information gain
#         information_gain = entropy_total - (len(x_left)/len(x)*entropy_left + len(x_right)/len(x)*entropy_right)
#         if information_gain > max_information_gain:
#             max_information_gain = information_gain
#             final_x_left = x_left
#             final_y_left = y_left
#             final_x_right = x_right
#             final_y_right = y_right
#             final_feature_index = feature_index
#             final_split_point = split_point

#     # Handle case where no split improves information gain
#     if final_x_left is None or final_x_right is None:
#         no_improvement_flag = True
#         return None, None, None, None, None, None, no_improvement_flag
#     else:
#         no_improvement_flag = False
#         return final_x_left, final_y_left[:, np.newaxis], final_x_right, final_y_right[:, np.newaxis], final_feature_index, final_split_point, no_improvement_flag


# def compute_entropy(y):
#     """Compute entropy of given dataset, x for features and y for labels

#     Args:
#         y (np.ndarray): NumPy array with shape (N, ) with integers from 1 to C, where C is the number of classes

#     Returns:
#         entropy (float): Entropy of given dataset
#     """
#     class_labels, class_counts = np.unique(y, return_counts=True)
#     n_instances = len(y)

#     entropy = 0.
#     for count in class_counts:
#         entropy = entropy - (count/n_instances * np.log2(count/n_instances))

#     return entropy


# def decision_tree_learning(dataset, depth=None, node_depth=0):
#     """Recursively build continuous-valued decision tree based on input dataset

#     Args:
#         dataset (np.ndarray): Input dataset with features and labels
#         node_depth (int): Current depth of the node in the tree
#         depth (int): Depth parameter to limit visualization of tree

#     Returns:
#         dict: Decision tree nodes in nested dictionary format
#     """

#     # Split dataset into features and labels
#     x, y = dataset[:, :-1], dataset[:, -1]
#     entropy_total = compute_entropy(y)

#     # Terminate if node is pure
#     if entropy_total == 0.:
#         return {"feature_to_split": None,
#                 "split_point": None,
#                 "node_depth": node_depth,
#                 "depth": depth,
#                 "labels": y,
#                 "leaf": True,
#                 "left": None,
#                 "right": None}
#     else:
#         split_points = compute_split_point(x)
#         x_left, y_left, x_right, y_right, feature_index, split_point, no_improvement_flag = split_dataset(x, y, split_points, entropy_total)
        
#         if no_improvement_flag:
#             return {"feature_to_split": None,
#                     "split_point": None,
#                     "node_depth": node_depth,
#                     "depth": depth,
#                     "labels": y,
#                     "leaf": True,
#                     "left": None,
#                     "right": None}
#         else:
#             dataset_left = np.hstack((x_left, y_left))
#             dataset_right = np.hstack((x_right, y_right))

#             return {"feature_to_split": feature_index,
#                     "split_point": split_point,
#                     "node_depth": node_depth,
#                     "depth": depth,
#                     "labels": y,
#                     "leaf": False,
#                     "left": decision_tree_learning(dataset_left, depth, node_depth+1),
#                     "right": decision_tree_learning(dataset_right, depth, node_depth+1)}





# def search_nodes_to_prune(node, directions_to_parents_to_prune, current_direction=""):
#     """Recursively search decision tree for parent nodes connected to 2 leaf nodes.
#     Return array of directions to reach these nodes from root.

#     Args:
#         node (_type_): _description_
#         validation_dataset (_type_): _description_
#     """

#     # Check if node is None (i.e. invalid child nodes of a leaf node) or leaf node. Do nothing (terminate)
#     if node is None or node.get("leaf") == True:
#         return None

#     # Check current node if left and right child nodes are leaves. If yes (parent node), append direction to it.
#     if node["left"].get("leaf") == True and node["right"].get("leaf") == True:
#         directions_to_parents_to_prune.add(current_direction)
#         return None

#     # Fire recursive search_and_prune_node down left and right child nodes
#     search_nodes_to_prune(node["left"], directions_to_parents_to_prune, current_direction+"l")
#     search_nodes_to_prune(node["right"], directions_to_parents_to_prune, current_direction+"r")

#     return None


