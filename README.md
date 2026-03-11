# Decision Tree Classifier from Scratch

A binary decision tree classifier built from scratch using only NumPy. Trains and evaluates with stratified k-fold cross-validation, supports reduced-error pruning via nested CV, and visualises the learned tree with Matplotlib.

Applied to WiFi-based indoor localisation — predicting which of 4 rooms a person is in based on signal strength measurements from 7 access points.

## Highlights

- Pure NumPy implementation — no scikit-learn
- Information gain (entropy) splits using the median value per feature
- Stratified 10-fold cross-validation
- Reduced-error pruning via nested cross-validation
- Tree visualisation exported as PNG

## Project Structure

```
├── main.py                  # Entry point: trains, evaluates, visualises
├── requirements.txt
├── src/
│   ├── dataset.py           # Data loading and stratified k-fold splitting
│   ├── model.py             # Decision tree learning, entropy/IG, pruning
│   ├── evaluation.py        # Confusion matrix, accuracy, precision, recall, F1
│   └── visuals.py           # Matplotlib tree plotting
└── wifi_db/
    ├── clean_dataset.txt    # 2,000 samples
    └── noisy_dataset.txt    # 1,999 samples
```

## Usage

```bash
pip install -r requirements.txt
python main.py wifi_db/clean_dataset.txt
# or
python main.py wifi_db/noisy_dataset.txt
```

Output:
1. Average accuracy, recall, precision, and F1 across folds
2. Saved tree visualisation (e.g. `wifi_db/clean_dataset_tree.png`)
3. Post-pruning metrics from nested CV

## Dataset Format

Whitespace-separated `.txt` file — first K-1 columns are float features, last column is the integer class label:

```
-64 -56 -61 -66 -71 -82 -81 1
```

## Academic Context

Developed as coursework for **COMP70050 Introduction to Machine Learning** at Imperial College London (MSc AI).

## Contributors

- Ethan Chia Wei Fong
- Benjamin Ang
- Catalina Tan
- Jia Yi Huang
