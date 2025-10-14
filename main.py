import numpy as np
from numpy.random import default_rng

# Task 1 
def load_dataset(file_name):
    with open(file_name, "r") as file:
        data = np.loadtxt(file)
        x = data[:, 0:-1]
        y = data[:, -1]
    return x, y

# Task 2 
def train_test_split(x, y, test_proportion, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    return (x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    x, y = load_dataset("wifi_db/clean_dataset.txt")