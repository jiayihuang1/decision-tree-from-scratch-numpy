## Continuous Decision Tree from Scratch (NumPy)

Builds a decision tree classifier from scratch using NumPy only, trains/evaluates it with cross-validation, optionally prunes it with nested CV, and visualizes the learned tree with Matplotlib. The datasets are simple whitespace-separated text files.

## Highlights

- Pure NumPy implementation of a binary decision tree
- Information gain (entropy) splits, using the median value per feature
- Stratified K-fold cross-validation for evaluation
- Optional pruning via nested cross-validation
- Tree visualization saved as a PNG

## Project structure

```
intro-to-ml-cw1/
├── main.py                      # Entry point: trains, evaluates, visualizes
├── requirements.txt             # Python dependencies
├── src/
│   ├── dataset.py              # Loading dataset and stratified k-fold split
│   ├── evaluation.py           # Confusion matrix and metrics
│   ├── model.py                # Decision tree learning, entropy/IG, pruning
│   └── visuals.py              # Matplotlib tree plotting
└── wifi_db/
		├── clean_dataset.txt
		└── noisy_dataset.txt
```

## Dataset format

- File: whitespace-separated values in a plain text `.txt`
- Shape: N rows by K columns
	- First K-1 columns are features (floats)
	- Last column is the integer class label

Example:

```
1.2 3.4 0
2.1 1.7 1
...
```

## Quick start

Run with an included dataset. If no path is passed, you’ll be prompted in the terminal.

```zsh
python main.py wifi_db/clean_dataset.txt
# or
python main.py wifi_db/noisy_dataset.txt
```

What it does:

1. Loads the dataset as a NumPy array
2. Performs stratified K-fold (K=10) cross-validation without pruning and reports average metrics
3. Trains one tree (from the last fold in the loop) and saves a visualization PNG next to the dataset
4. Runs nested CV to prune the tree and reports average accuracy after pruning

Outputs:

- Average accuracy, recall, precision, F1 (from the average confusion matrix)
- Path to the saved tree plot, e.g. `wifi_db/clean_dataset_tree.png`


## Contributors

- Ethan Chia Wei Fong
- Benjamin Ang
- Cataline Tan
- Jia Yi Huang
