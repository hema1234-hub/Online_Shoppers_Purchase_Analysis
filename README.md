# Online_Shoppers_Purchase_Analysis
Foundations of Data Science case study analyzing online shopper behavior and purchase conversion using the Kaggle Online Shoppers Purchasing Intention dataset.# Online Shopper Purchase Analysis

## Research question
Which session characteristics are associated with an online purchase?

## Dataset
[Kaggle: Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
The dataset contains session summaries and a purchase outcome called `Revenue`.

## Results
- Sessions: 12,330
- Purchase sessions: 1,908 (15.47%)
- Non-purchase sessions: 10,422 (84.53%)
- The outcome is imbalanced: non-purchase sessions are much more common.

## Model Results
The logistic regression model was evaluated on 2,441 test sessions and achieved 63% accuracy.
| Outcome | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No purchase (`False`) | 0.92 | 0.62 | 0.74 | 2,059 |
| Purchase (`True`) | 0.25 | 0.69 | 0.37 | 382 |
The model identified 69% of actual purchase sessions, but only 25% of its purchase predictions were correct. It produced a convergence warning, so these results are preliminary and should be updated after rerunning the model without that warning.

## Limitations
The data contains session summaries, not chronological cart events, so it cannot measure cart abandonment. The results show associations and do not prove what causes a purchase.
**Note:** The model produced a convergence warning, so these scores are preliminary.
