# Online Shopper Purchase Analysis

**Foundations of Data Science case study** analyzing online shopper behavior and purchase conversion using the Kaggle *Online Shoppers Purchasing Intention* dataset. The project predicts whether a browsing session ends in a purchase (`Revenue = True`) and identifies session characteristics associated with conversion. [kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

## Problem Statement

Online stores need to understand which browsing patterns lead to completed purchases so they can improve product pages, targeting, and conversion rates. This project uses session-level features—such as product-page activity, bounce/exit rates, page value, visitor type, and timing—to classify sessions as purchase or no-purchase sessions. [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

**Research question:** Which session characteristics are associated with an online purchase?

## Introduction

The dataset contains summaries of 12,330 e-commerce sessions collected over one year. Each row represents one user session, with 10 numerical and 8 categorical attributes; `Revenue` is the binary target variable indicating whether the session ended in a transaction. [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

The target is highly imbalanced: only 1,908 sessions (15.47%) ended in a purchase, while 10,422 sessions (84.53%) did not. This imbalance means accuracy alone is not enough; precision, recall, and F1-score for the purchase class are more informative. [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## Dataset

**Source:** [Kaggle — Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset) [kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

| Feature | Description |
|---|---|
| `Administrative`, `Administrative_Duration` | Number of account/admin pages visited and time spent on them.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `Informational`, `Informational_Duration` | Number of informational pages visited and time spent on them.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `ProductRelated`, `ProductRelated_Duration` | Number of product-related pages visited and time spent on them.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `BounceRates` | Average bounce rate for pages visited in the session.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `ExitRates` | Average exit rate for pages visited in the session.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `PageValues` | E-commerce value associated with the pages visited.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| `SpecialDay` | Closeness of the visit date to a special day.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| `Month` | Month of the session.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Encoded technical and traffic-source attributes.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| `VisitorType` | New, returning, or other visitor.  [stat.cmu](https://www.stat.cmu.edu/capstoneresearch/fall2022/315files_f22/team4.html) |
| `Weekend` | Whether the session occurred on a weekend.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| `Revenue` | Target: `True` if the session ended in a purchase, otherwise `False`.  [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |

## Project Overview

1. Load and inspect the dataset.
2. Clean data types and handle categorical variables.
3. Perform exploratory data analysis on purchase vs. non-purchase sessions.
4. Split data into training and test sets.
5. Scale numerical features and encode categorical features.
6. Train a logistic regression classifier.
7. Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.

## Model Results

The logistic regression model was evaluated on 2,441 test sessions and achieved **63% accuracy**.

| Outcome | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No purchase (`False`) | 0.92 | 0.62 | 0.74 | 2,059 |
| Purchase (`True`) | 0.25 | 0.69 | 0.37 | 382 |

The model identified **69% of actual purchase sessions**, but only **25% of its purchase predictions were correct**. This indicates that the model is useful for finding many potential buyers, but it generates many false-positive purchase predictions. [archive-beta.ics.uci](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

The model produced a convergence warning, so the reported scores are preliminary. Rerun the model with increased `max_iter`, proper scaling, and possibly class weighting before treating these results as final.

## How to Run

```bash
git clone <your-repository-url>
cd Online_Shoppers_Purchase_Analysis

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Open the notebook and run all cells. Place the dataset file, `online_shoppers_purchasing_intention.csv`, in the project folder or update the file path in the notebook.

## Repository Structure

```text
Online_Shoppers_Purchase_Analysis/
│
├── data/
│   └── online_shoppers_purchasing_intention.csv
│
├── notebooks/
│   └── online_shopper_purchase_analysis.ipynb
│
├── src/
│   └── analysis.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Future Improvements

- Resolve the logistic regression convergence warning using feature scaling and a higher `max_iter`.
- Apply `class_weight="balanced"` or resampling methods such as SMOTE to improve purchase-class performance.
- Compare logistic regression with Random Forest, XGBoost, and decision-tree models.
- Tune hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
- Analyze feature importance and SHAP values to explain which variables most influence purchase prediction.
- Use precision-recall curves and ROC-AUC, since the classes are imbalanced.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

## Author

**Poripireddi Hema**  

## License

This project is licensed under the MIT License.
