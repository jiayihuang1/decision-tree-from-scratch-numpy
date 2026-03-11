# Decision Tree Classifier from Scratch: Training, Pruning & Evaluation

A pure NumPy implementation of a binary decision tree classifier with entropy-based splitting, multi-pass reduced-error pruning, and rigorous nested cross-validation. Applied to WiFi signal strength data for indoor room localisation — a real-world classification task relevant to warehouse robotics, retail analytics, and smart building navigation.

**Built without scikit-learn or ML libraries** — every algorithm (entropy, information gain, tree construction, pruning, stratified k-fold, confusion matrix, precision/recall/F1) is implemented from first principles.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.7+ |
| **Numerical Computing** | NumPy (vectorised operations, array slicing, statistical aggregation) |
| **Visualisation** | Matplotlib (recursive tree plotting) |

---

## Key Algorithms

| Stage | Method | Detail |
|---|---|---|
| Splitting Criterion | Entropy + Information Gain | H(S) = -Σ p·log₂(p); IG = H(parent) - weighted H(children) |
| Split Selection | Median-value per feature | For each feature, split at median; select feature with max IG |
| Tree Construction | Recursive binary partitioning | Stops at pure nodes (entropy = 0) or zero IG (majority vote) |
| Pruning | Multi-pass reduced-error pruning | Bottom-up leaf collapsing; iterates until validation accuracy stops improving; auto-rollback on degradation |
| Evaluation | Stratified k-fold CV | Class-proportional splits ensure balanced folds; nested CV separates pruning from testing |
| Metrics | Confusion matrix from scratch | Per-class TP/FP/FN for precision, recall, F1; no sklearn dependency |

---

## Architecture & Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DECISION TREE TRAINING PIPELINE                    │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐             │
│  │  Dataset  │───>│  Stratified   │───>│   Decision     │             │
│  │  Loading  │    │  K-Fold Split │    │   Tree         │             │
│  │  (NumPy)  │    │  (K=10)      │    │   Learning     │             │
│  └──────────┘    └──────────────┘    └───────┬────────┘             │
│                                              │                       │
│                                              v                       │
│  ┌───────────────────────────────────────────────────────┐          │
│  │              Recursive Tree Construction               │          │
│  │  ┌─────────────────┐    ┌───────────────────────────┐ │          │
│  │  │  For each node:  │    │  Split criterion:          │ │          │
│  │  │  Compute entropy │───>│  argmax Information Gain   │ │          │
│  │  │  H(S) = -Σp·log₂p│    │  over median-split per    │ │          │
│  │  │                  │    │  feature                   │ │          │
│  │  └─────────────────┘    └───────────────────────────┘ │          │
│  └───────────────────────────┬───────────────────────────┘          │
│                              v                                       │
│  ┌─────────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │  Multi-Pass      │───>│  Nested CV   │───>│  Classification   │   │
│  │  Reduced-Error   │    │  (10 × 9 =   │    │  Metrics          │   │
│  │  Pruning         │    │  90 folds)   │    │  (per-class P/R/F1)│   │
│  └─────────────────┘    └──────────────┘    └───────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Results

| Metric | Clean Dataset | Noisy Dataset |
|---|---|---|
| **Accuracy (without pruning)** | 96.05% | 78.80% |
| **Accuracy (after pruning)** | 96.47% | 87.59% |
| **Pruning improvement** | +0.42% | +8.79% |

> Pruning shows its value on noisy data: **+8.79% accuracy gain** by removing overfitted branches, while maintaining performance on clean data.

---

## Project Structure

```
├── main.py                  # Pipeline orchestrator: train, evaluate, visualise, prune
├── requirements.txt
├── src/
│   ├── dataset.py           # Dataset loading and stratified k-fold splitting
│   ├── model.py             # Decision tree learning, entropy, IG, find_split, pruning
│   ├── evaluation.py        # Confusion matrix, accuracy, precision, recall, F1, prediction
│   └── visuals.py           # Recursive Matplotlib tree plotting with leaf positioning
└── wifi_db/
    ├── clean_dataset.txt    # 2,000 samples
    └── noisy_dataset.txt    # 1,999 samples
```

---

## Dataset

- **Format**: Whitespace-separated `.txt` files — N rows × K columns
- **Features**: First K-1 columns are WiFi signal strengths (floats, in dBm)
- **Labels**: Last column is integer room label (4 classes)
- **Domain**: Indoor localisation — classify which room a person occupies based on WiFi access point signal strengths

```
-64  -56  -61  -66  -71  -82  -81     1
-68  -57  -61  -65  -71  -85  -85     1
```

---

## Getting Started

```bash
git clone https://github.com/jiayihuang1/decision-tree-from-scratch-numpy.git
cd decision-tree-from-scratch-numpy

pip install -r requirements.txt

python main.py wifi_db/clean_dataset.txt
# or
python main.py wifi_db/noisy_dataset.txt
```

The pipeline executes four stages in sequence:

1. **Train & Evaluate** — Builds decision trees across 10 stratified folds; reports accuracy, precision, recall, F1
2. **Visualise** — Saves a tree plot as PNG next to the dataset
3. **Prune & Re-evaluate** — Runs nested CV (10 outer × 9 inner folds) with multi-pass reduced-error pruning
4. **Compare** — Reports metrics before and after pruning

---

## Academic Context

Developed as coursework for **COMP70050 Introduction to Machine Learning** at Imperial College London (MSc AI).

## Contributors

- Ethan Chia Wei Fong
- Benjamin Ang
- Catalina Tan
- Jia Yi Huang
