# Customer Purchase Prediction — End-to-End Machine Learning

## 1. Project Overview

### Objective

Build a machine learning model that predicts whether a customer will make a purchase within the next 30 days, using historical transaction data.

The project simulates a customer analytics use case in which a business wants to identify customers who are likely to purchase in the near future. These predictions could support customer segmentation, marketing campaigns, and customer relationship management.

### Business Problem

Retail businesses need to identify customers who are likely to return and purchase again. Historical transaction data can reveal behavioral patterns associated with future purchases, but a useful predictive model must generalize to future periods rather than simply memorize historical behavior.

This project explores whether customer purchase history can provide useful predictive signals and compares multiple classification algorithms using a chronological evaluation strategy.

### Dataset

**Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

The dataset contains transactions from a UK-based online retailer between December 2010 and December 2011.

Relevant columns include:

- `InvoiceNo`: transaction identifier.
- `StockCode`: product identifier.
- `Description`: product description.
- `Quantity`: quantity purchased.
- `InvoiceDate`: transaction timestamp.
- `UnitPrice`: unit price.
- `CustomerID`: customer identifier.
- `Country`: customer country.

The project focuses on customer-level purchase behavior rather than product-level recommendations.

---

## 2. Problem Formulation

### Prediction Task

This is a binary classification problem.

For each prediction date, the model uses a customer's historical purchasing behavior to predict whether that customer will make at least one valid purchase during the following 30 days.

The target is defined as:

- `1`: The customer makes at least one valid purchase during the next 30 days.
- `0`: The customer makes no valid purchase during that period.

The target window includes the prediction date and excludes the date 30 days later.

### Prediction Snapshots

Customer features and labels are generated at multiple monthly prediction dates:

- June 1, 2011
- July 1, 2011
- August 1, 2011
- September 1, 2011
- October 1, 2011
- November 1, 2011

Each snapshot represents the information available at the time a prediction would have been made.

Customers must have valid purchase history before the prediction date to be included in that snapshot.

### Feature Engineering

Three initial features were created for each customer.

| Feature | Definition | Interpretation |
|---|---|---|
| `recency` | Days since the customer's most recent purchase before the prediction date | Measures customer inactivity |
| `total_orders` | Number of unique invoices before the prediction date | Measures purchase frequency |
| `total_spent` | Total historical revenue before the prediction date | Measures cumulative spending |

Revenue is calculated as:

`revenue = Quantity × UnitPrice`

`CustomerID` is retained as an identifier but excluded from the model features.

These features provide a simple baseline representation of customer behavior. More advanced behavioral features may improve predictive performance in future iterations.

---

## 3. Data Preparation and Leakage Prevention

The original dataset contains cancelled transactions, negative quantities, missing customer identifiers, and incomplete product descriptions.

The preparation process includes:

- Excluding cancelled invoices identified by the `C` prefix.
- Excluding negative quantities from the valid-purchase modeling dataset.
- Excluding records without a usable customer identifier.
- Excluding records with missing product descriptions during the defined cleaning process.
- Calculating transaction revenue.
- Converting transaction timestamps into a consistent datetime representation.

These rules establish the transaction population used for feature engineering and target construction. In a production setting, cancellation handling and returns would require additional business-specific validation.

### Preventing Data Leakage

Leakage prevention is essential because the model is intended to predict future behavior.

The following rules were applied:

1. Customer features use only transactions strictly before each prediction date.
2. Target labels use transactions in the subsequent 30-day window.
3. `CustomerID` is not used as a predictive feature.
4. Data is divided chronologically rather than randomly.
5. The scaler for Logistic Regression and the initial Random Forest experiments is fitted on training data only.
6. The November test set is reserved for final evaluation, not model or threshold selection.

A customer can appear in multiple snapshots because each row represents that customer's state at a different prediction date. Duplicate customer-date combinations are not allowed.

---

## 4. Temporal Train, Validation, and Test Split

The combined dataset contains 19,733 customer-snapshot observations across six monthly prediction dates.

| Split | Prediction dates | Observations | Purpose |
|---|---|---:|---|
| Training | June–August 2011 | 8,826 | Fit model parameters |
| Validation | September–October 2011 | 6,933 | Compare models and select a classification threshold |
| Test | November 2011 | 3,974 | Estimate final performance on a later period |

The target distributions differ across periods, with the positive class becoming more prevalent in November.

This makes chronological evaluation especially relevant: it tests the model on a later period whose customer behavior may differ from the training period.

---

## 5. Evaluation Metrics

Because the objective is to identify future purchasers, accuracy alone is insufficient.

The project uses the following metrics:

- **Precision:** Proportion of predicted purchasers who actually purchase.
- **Recall:** Proportion of actual purchasers identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank purchasers above non-purchasers across classification thresholds.
- **Average Precision (PR-AUC):** Summary of the precision-recall curve, useful for assessing positive-class identification.
- **Accuracy:** Proportion of all predictions classified correctly.

Class `1` represents customers who purchase during the target window and is the primary class of interest.

### Classification Threshold

A classifier's predicted probability must be converted into a binary prediction using a threshold.

The default threshold of `0.5` can produce an undesirable precision-recall trade-off. Therefore, the threshold was selected by maximizing class 1 F1-score on the validation set.

The selected threshold must remain fixed during final test evaluation. Choosing a threshold using test labels would introduce test-set information into model selection and produce an optimistically biased estimate.

---

## 6. Models and Experiments

### Logistic Regression

Logistic Regression was used as the initial classification baseline.

The numerical features were standardized using `StandardScaler`, fitted only on training data.

At the default threshold of `0.5`, validation accuracy was approximately 76.16%, but recall for purchasers was only 29.73%.

Lowering the threshold increased recall and improved the class 1 F1-score.

**Best validation threshold:** approximately `0.2136`

**Validation F1-score:** approximately `0.5270`

### Random Forest

A Random Forest was tested to capture nonlinear relationships and interactions between customer features.

The initial configuration used:

- `max_depth=6`
- `n_estimators=15`
- `min_samples_leaf=3`

Its ROC-AUC was approximately `0.7290`, compared with `0.7260` for Logistic Regression.

Threshold tuning produced a best validation F1-score of approximately `0.5294`, a marginal improvement over Logistic Regression.

### HistGradientBoostingClassifier

Histogram-based Gradient Boosting was tested as a third model family.

The initial configuration used:

- `max_iter=1000`
- `max_depth=3`
- `min_samples_leaf=500`

This configuration achieved a validation ROC-AUC of approximately `0.7222`.

The minimum leaf size was then varied while the other parameters remained fixed. The best ROC-AUC among the tested configurations occurred at `min_samples_leaf=81`.

The resulting validation ROC-AUC was approximately `0.7312`.

The selected threshold was approximately `0.2518`, producing a validation F1-score of approximately `0.5305`.

### Model Selection

HistGradientBoostingClassifier was selected as the provisional final model because it achieved the highest validation ROC-AUC and a marginally higher validation F1-score among the three tested models.

The improvement over Random Forest was small, so the results do not establish that Gradient Boosting is substantially superior. The selection reflects the best observed validation performance, not proof of a universally better algorithm.

---

## 7. Final Test Results

The selected Gradient Boosting model was evaluated on the November test set using the threshold selected on validation.

**Selected threshold:** `0.251797`

| Metric | November test result |
|---|---:|
| Accuracy | 68.67% |
| ROC-AUC | 0.7300 |
| Average Precision | 0.6308 |
| Class 1 precision | 53% |
| Class 1 recall | 61% |
| Class 1 F1-score | 0.57 |
| Class 0 F1-score | 0.75 |

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| | Predicted: No purchase | Predicted: Purchase |
|---|---:|---:|
| Actual: No purchase | 1,915 | 718 |
| Actual: Purchase | 527 | 814 |

### Interpretation

On the November test set:

- The model correctly identified 814 customers who purchased.
- It missed 527 actual purchasers.
- It flagged 718 customers who did not purchase.
- Its purchaser recall was approximately 60.7%.
- Its purchaser precision was approximately 53.1%.

The model therefore identifies a meaningful proportion of future purchasers, but a substantial number of its positive predictions are false positives.

Whether this performance is commercially useful depends on campaign costs, conversion value, and the consequences of contacting customers who do not purchase.

### Baseline Comparison

A naive classifier that predicts no purchase for every customer achieves test accuracy of approximately 66.26%, reflecting the proportion of non-purchasers in November.

The selected model achieves 68.67% accuracy, an improvement of approximately 2.41 percentage points.

The majority-class baseline has zero recall for purchasers, whereas the selected model identifies approximately 60.7% of them.

Accuracy improvement is modest, but the ability to identify likely purchasers provides a different and potentially more useful business function.

---

## 8. Limitations

Several limitations remain.

**Limited feature set:** The model uses only recency, order count, and cumulative spending. Purchase intervals, spending trends, average order value, product diversity, and country-specific behavior may contain additional predictive information.

**Single test period:** Final evaluation uses November 2011 only. Performance in one later period does not establish stability across multiple future periods.

**Changing class prevalence:** The proportion of purchasers differs across the chronological splits. This can affect precision, recall, and F1-score even when ROC-AUC remains similar.

**Threshold objective:** Maximizing F1 treats precision and recall symmetrically. A real marketing decision should account for campaign costs, customer value, and the business cost of false positives and false negatives.

**Limited model search:** Only three model families were evaluated, with relatively limited hyperparameter exploration.

**Historical dataset:** The data comes from 2010–2011. Results demonstrate a modeling workflow, not expected performance on a modern retailer's current customer base.

---

## 9. Next Steps

The initial modeling milestone is complete. Further work should focus on improving the quality and practical relevance of the solution.

1. **Feature engineering:** Add average order value, purchase frequency, time since previous orders, and spending trends.
2. **Business-oriented threshold selection:** Estimate the value of successful conversions and the cost of marketing contacts, then select a threshold using expected business value.
3. **Robust temporal validation:** Evaluate across additional rolling prediction periods where sufficient data is available.
4. **Model interpretation:** Examine feature importance and, where appropriate, use permutation importance or SHAP to understand the drivers of predictions.
5. **Reproducibility:** Package preprocessing and model inference into a consistent pipeline.
6. **Deployment:** Expose predictions through an API and containerize the application.
7. **Monitoring:** Track prediction quality, class prevalence, and feature drift when new data becomes available.

## 10. Conclusion

This project establishes an end-to-end workflow for customer purchase prediction using transaction-level retail data.

It covers data cleaning, customer-level feature engineering, temporal target construction, leakage prevention, model comparison, threshold selection, and final out-of-time evaluation.

HistGradientBoostingClassifier achieved a test ROC-AUC of approximately `0.7300` and a purchaser F1-score of approximately `0.57` using a validation-selected threshold.

The results indicate that historical customer behavior contains predictive information, although performance remains moderate and further feature engineering and business-oriented evaluation are necessary.

The primary outcome is not just a fitted classifier, but a reproducible framework for evaluating predictive models under a realistic temporal forecasting setup.
