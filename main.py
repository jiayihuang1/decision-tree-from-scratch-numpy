import sys
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from src.evaluation import evaluate, generate_classification_metrics, compute_accuracy, predict
from src.dataset import load_dataset, train_test_k_fold_split
from src.model import decision_tree_learning, prune_n_parses
from src.visuals import plot_tree

if __name__ == "__main__":
    # Set constant random seed for reproducibility
    seed = 60012
    rg = default_rng(seed)

    # Request user for file path to dataset
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter path to dataset.txt file (e.g., 'wifi_db/clean_dataset.txt'): ")

    # Load dataset and split into train-test in stratified manner
    dataset = load_dataset(dataset_path)
    K = 10
    train_test_k_folds = train_test_k_fold_split(dataset, K, rg)

    # Initialize array to aggregate confusion matrices
    total_confusion_matrix = np.zeros((len(np.unique(dataset[:, -1])), len(np.unique(dataset[:, -1]))))

    # Train decision tree and evaluate on each fold
    print(f"Training and running simple CV on trees without pruning for {K} folds, please wait...")
    for k, train_test_fold in enumerate(train_test_k_folds):
        train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
        decision_tree, max_depth = decision_tree_learning(train_dataset)
        confusion_matrix = evaluate(test_dataset, decision_tree)
        total_confusion_matrix += confusion_matrix

    # Compute average confusion matrix. Then, compute metrics from confusion matrix
    average_confusion_matrix = total_confusion_matrix / K
    accuracy, recall, precision_rate, f1 = generate_classification_metrics(average_confusion_matrix)
    print(f"\nAverage metrics for tree (without pruning)")
    print(f"Accuracy: {accuracy}%")
    print(f"Recall per class: {recall}")
    print(f"Precision rate per class: {precision_rate}")
    print(f"F1 score per class: {f1}")
    print(f"Average confusion matrix:\n{average_confusion_matrix}")

    # Plot and save visualization of a trained decision tree
    ax = plot_tree(decision_tree)
    plt.tight_layout()
    plt.savefig(f"{dataset_path.split('.')[0]}_tree.png", dpi=300, bbox_inches='tight')
    print(f"Tree visualization saved as '{dataset_path.split('.')[0]}_tree.png'")
    plt.close()
    
    # Setup dataset for nested cross validation: [test_1, [[train_1, val_1], [train_2, val_2], ...]]
    train_test_val_folds = []
    for train_test_fold in train_test_k_folds:
        train_dataset, test_dataset = train_test_fold[0], train_test_fold[1]
        train_test_val_folds.append([test_dataset, train_test_k_fold_split(train_dataset, K-1, rg)])

    # Train and prune decision tree. Then evaluate on test dataset
    nested_cv_accuracies = []

    print(f"Training, pruning and running nested CV on trees for {K*(K-1)} folds, please wait...")
    for train_test_val_fold in train_test_val_folds:
        test_dataset = train_test_val_fold[0]

        for train_val_fold in train_test_val_fold[1]:
            train_dataset, val_dataset = train_val_fold[0], train_val_fold[1]
            decision_tree, max_depth = decision_tree_learning(train_dataset)

            pruned_tree = prune_n_parses(decision_tree, train_dataset, val_dataset)
            y_prediction = predict(pruned_tree, test_dataset)
            accuracy = compute_accuracy(test_dataset[:, -1], y_prediction)
            nested_cv_accuracies.append(accuracy)

    # Compute average accuracy across all nested CV folds
    nested_cv_average_accuracy = sum(nested_cv_accuracies)/len(nested_cv_accuracies)

    print(f"\nAverage accuracy after pruning with nested CV: {nested_cv_average_accuracy*100:.2f}%")

