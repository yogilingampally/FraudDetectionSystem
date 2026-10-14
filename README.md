# Credit Card Fraud Detection

An XGBoost fraud detector with a Streamlit app. Score a real transaction, upload a CSV of transactions, or enter your own values, and see how the model decides and why.

**Live demo:** [add your Streamlit link here after deploying]

## What the app does

- **Sample transaction:** loads a real row from the held-out test set and shows the model's decision next to the true label, including cases the model gets wrong.
- **Upload CSV:** scores every row, shows the riskiest first, and lets you download the results. If the file has a `Class` column, it also counts caught frauds, misses and false alarms.
- **Manual entry:** start from a typical legit or fraud profile, then change Amount, Time or any PCA feature.
- **Explanations:** a bar chart of the top feature contributions (TreeSHAP) for each prediction.
- **Threshold slider:** change the decision threshold and see the precision and recall trade-off.

## Results (held-out test set)

Model: XGBoost, decision threshold 0.70.

| Metric | Value |
|---|---|
| Precision | 0.92 |
| Recall | 0.76 |
| F1 score | 0.83 |
| ROC-AUC | 0.96 |
| PR-AUC | 0.81 |

On 56,746 test transactions (95 fraud) the model caught 72 frauds, missed 23, and raised 6 false alarms among 56,651 legitimate transactions.

Fraud is only about 0.17% of transactions, so accuracy (99.95%) is not a useful headline number. Precision, recall and PR-AUC are the metrics that matter here.

## How the model was built

- **Data:** the public credit card fraud dataset (European cardholders, September 2013). Features V1 to V28 are anonymized PCA components, plus Time and Amount.
- **Cleaning:** 1,081 duplicate rows were removed before splitting, so the same transaction cannot appear in both training and test data.
- **Split:** stratified 60% train, 20% validation, 20% test.
- **Imbalance:** handled with class weights (`class_weight="balanced"` and XGBoost's `scale_pos_weight`), with no resampling.
- **Models compared:** Logistic Regression, Decision Tree, Random Forest and XGBoost.
- **Selection:** the model and threshold were chosen on the validation set (recall at least 0.80, then best F1 and precision). The test set was used once, for the numbers above.

## Limitations

- Recall was 0.86 on the validation set and 0.76 on the test set. Each split has only about 95 fraud cases, so these numbers are noisy.
- V1 to V28 are anonymized, so the explanation chart shows which hidden signals mattered but not what they mean.
- The data covers two days in 2013. This is a portfolio project, not a production fraud system.

## Run it locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

To retrain the model, download the dataset as `data/creditcard.csv` (it is not included in this repo because of its size) and run:

```
python train_model.py
```

## Project structure

```
app.py                  Streamlit app
train_model.py          Training, comparison and model saving
requirements.txt        Pinned package versions
models/                 Saved model, threshold, feature list, metrics and sample transactions
```
