# CustomerChurn_Prediction

This project identifies customers who are likely to leave the bank (churn) using a Random Forest Classifier trained on the Churn Modelling Dataset.

## Files

- `churn.py`: Main Python code with all preprocessing, training, and evaluation steps.
- `Churn_Modelling.csv`: Dataset used (must be stored locally, and path should be updated as needed).


## Objective

To build a machine learning classification model that predicts whether a customer will exit the bank, based on personal and account details like balance, tenure, credit score, geography, and more.


## Skills Used

- Data Cleaning: Dropping irrelevant columns.
- Categorical Encoding: One-Hot Encoding (`Geography`, `Gender`).
- Modeling: Random Forest Classifier with Scikit-learn.
- Evaluation: Accuracy score and confusion matrix.
- Feature Importance: Understanding which variables most impact customer churn.


## Summary of Preprocessing

- Dropped unhelpful columns: `RowNumber`, `CustomerId`, and `Surname`.
- Encoded categorical columns using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity.
- Defined feature set (`X`) and target (`y`), where `y = Exited` (1 if customer churned, 0 otherwise).
- Split data into training and testing sets (80/20 split).
- Trained a Random Forest Classifier on training data.
- Evaluated model using accuracy and confusion matrix.
- Visualized feature importance to see which features matter most (e.g., Credit Score, Age, Balance).


## Visual Outputs

- Confusion Matrix: Shows model predictions vs actual results.
- Feature Importance Bar Chart: Ranks input variables based on how much they affect the prediction.


##  How to Run

1. Ensure the `Churn_Modelling.csv` file is saved locally.
2. Update the path in the script if needed:

```python
df = pd.read_csv(r"your\local\path\Churn_Modelling.csv")
