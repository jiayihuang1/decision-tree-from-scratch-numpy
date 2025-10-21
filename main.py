import numpy as np
from numpy.random import default_rng

from src.dataset import load_dataset, train_test_k_fold_split
from src.model import decision_tree_learning, prune_n_parses
from src.evaluation import evaluate, generate_classification_metrics, compute_accuracy, predict


if __name__ == "__main__":

    # Set constant random seed for reproducibility
    seed = 60012
    rg = default_rng(seed)

    # Load dataset and split into train-test in stratified manner
    dataset = load_dataset("wifi_db/noisy_dataset.txt")
    K = 10
    train_test_k_folds = train_test_k_fold_split(dataset, K, rg)

    total_confusion_matrix = np.zeros((len(np.unique(dataset[:, -1])), len(np.unique(dataset[:, -1]))))
    # Train decision tree
    for k, train_test_fold in enumerate(train_test_k_folds):
        train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
        decision_tree, max_depth = decision_tree_learning(train_dataset)
        confusion_matrix = evaluate(test_dataset, decision_tree)
        total_confusion_matrix += confusion_matrix
    
    average_confusion_matrix = total_confusion_matrix / K
   # Compute classification metrics from the total confusion matrix
    accuracy, recall, precision_rate, F1 = generate_classification_metrics(average_confusion_matrix)
    print("\nAverage metrics for tree (without pruning)")
    print("Accuracy: ", accuracy)
    print("Recall per class: ", recall)
    print("Precision rate per class: ", precision_rate)
    print("F1 score per class: ", F1)

    # Setup dataset for nested cross validation: [test_1, [[train_1, val_1], [train_2, val_2], ...]]
    total_nested_cv_accuracies = []
    
    train_test_val_folds = []
    for train_test_fold in train_test_k_folds:
        train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
        train_test_val_folds.append([test_dataset, train_test_k_fold_split(train_dataset, 9, rg)])
    
    # Train and prune decision tree. Then evaluate on test dataset
    nested_cv_test_results = []

    for train_test_val_fold in train_test_val_folds:
        test_dataset = train_test_val_fold[0]

        for train_val_fold in train_test_val_fold[1]:
            train_dataset, val_dataset = train_val_fold[0], train_val_fold[1]
            decision_tree, max_depth = decision_tree_learning(train_dataset)

            # prune each decision tree over maximum of n_parses, evaluate accuracy of pruned tree
            pruned_tree = prune_n_parses(decision_tree, train_dataset, val_dataset)
            y_prediction = predict(pruned_tree, test_dataset)
            accuracy = compute_accuracy(test_dataset[:, -1], y_prediction)
            total_nested_cv_accuracies.append(accuracy)

    # calculate average accuracy across all nested CV folds
    nested_cv_average_accuracy = sum(total_nested_cv_accuracies)/len(total_nested_cv_accuracies)

    print("\nAverage accuracy after pruning with nested CV: ", nested_cv_average_accuracy)











# if __name__ == "__main__":

#     # Set constant random seed for reproducibility
#     seed = 60012
#     rg = default_rng(seed)

#     # Load dataset and split into train-test in stratified manner
#     dataset = load_dataset("wifi_db/noisy_dataset.txt")
#     train_test_k_folds = train_test_k_fold_split(dataset, 10, rg)

#     # Train decision tree
#     for k, train_test_fold in enumerate(train_test_k_folds):
#         train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
#         decision_tree = decision_tree_learning(train_dataset)

#         evaluation_results = evaluate(test_dataset, decision_tree)
#         # print(f"Confusion Matrix for fold {k}:\n", evaluation_results["confusion_matrix"])
#         # print(f"Accuracy for fold {k}:", evaluation_results["accuracy"])
#         # print(f"Precisions for fold {k}:", evaluation_results["precisions"])
#         # print(f"Macro Precision for fold {k}:", evaluation_results["macro_precision"])
#         # print(f"Recalls for fold {k}:", evaluation_results["recalls"])
#         # print(f"Macro Recall for fold {k}:", evaluation_results["macro_recall"])
#         # print(f"F1-scores for fold {k}:", evaluation_results["f1s"])
#         # print(f"Macro F1-score for fold {k}:", evaluation_results["macro_f1"])

#     # Setup dataset for nested cross validation: [test_1, [[train_1, val_1], [train_2, val_2], ...]]
#     train_test_val_folds = []
#     for train_test_fold in train_test_k_folds:
#         train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
#         train_test_val_folds.append([test_dataset, train_test_k_fold_split(train_dataset, 9, rg)])
    
#     # Train and prune decision tree. Then evaluate on test dataset
#     nested_cv_test_results = []

#     for train_test_val_fold in train_test_val_folds:
#         test_dataset = train_test_val_fold[0]

#         for train_val_fold in train_test_val_fold[1]:
#             train_dataset, val_dataset = train_val_fold[0], train_val_fold[1]
#             decision_tree = decision_tree_learning(train_dataset)

#             directions_to_parents_to_prune = set()
#             search_nodes_to_prune(decision_tree, directions_to_parents_to_prune)
#             print(list(directions_to_parents_to_prune)[0])



#             print(decision_tree["left"]["right"]["right"]["left"]["left"]["left"]["left"]["right"]["right"])
#             raise Exception("Stop")
#             # TODO: pruned_tree = search_and_prune_node(decision_tree, val_dataset)
#             # TODO: nested_cv_test_results.append(evaluate(test_dataset, pruned_tree))



#     # print(decision_tree)
#         # print(predict(decision_tree, test_dataset[:, :-1]))
