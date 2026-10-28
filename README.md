# Shopflow: Customer Purchase Prediction

A machine learning project that predicts whether a customer will buy something in the next 30 days, using only their past transactions.

I built it to practice the full workflow on a real dataset: cleaning, features, a time-based split, model comparison, picking a threshold, and testing on a later period. The goal was to do it honestly, so this README includes what didn't work and where the model is weak.

## The question

A shop wants to know which customers are likely to buy soon, so it can decide who to contact. For each customer, on a given date, the model uses their history up to that date and predicts: **will they buy at least once in the next 30 days?** (1 = yes, 0 = no)

## Data

[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) from the UCI Machine Learning Repository. It has transactions from a UK online shop between December 2010 and December 2011.

I keep only rows with a positive `Quantity` and a known `CustomerID`:

- 541,909 raw rows become 397,924 usable rows from 4,339 customers.
- Cancelled invoices start with `C` and always have a negative quantity, so the quantity rule removes all of them (9,288 rows).
- Rows without a customer ID can't be tied to anyone, so they're dropped.
- Returns are not subtracted from a customer's spending, they are just left out.
- 40 rows with a unit price of 0 are still in the data. I didn't filter them.

## How the data is set up

Instead of one row per customer, I take **monthly snapshots**. On the 1st of each month from June to November 2011, I look at every customer who had bought before that date, compute their features using only earlier purchases, and check whether they bought in the next 30 days. A customer can appear in several snapshots, once per month.

The split is by date, not random, because the model should be tested on a period it hasn't seen:

| Split | Snapshots | Rows | Share who bought |
|---|---|---:|---:|
| Train | Jun, Jul, Aug | 8,826 | 25.4% |
| Validation | Sep, Oct | 6,933 | 28.0% |
| Test | Nov | 3,974 | 33.7% |

To avoid leaking the future into the model:

- Features only use purchases **strictly before** the snapshot date.
- The target only looks at the 30 days **starting on** the snapshot date. `build_target` raises an error if that window isn't fully in the data.
- `CustomerID` is never a feature.
- The validation set is used to compare models and choose the threshold. The test set (November) was used only once, at the end.

## Features

Five numbers per customer, as of the snapshot date:

| Feature | What it is |
|---|---|
| `recency` | days since their last purchase |
| `total_orders` | number of distinct invoices |
| `total_spent` | total money spent (`Quantity × UnitPrice`) |
| `avg_order` | `total_spent / total_orders` |
| `purchased_months` | number of different months they bought in |

I started with the first three and added the last two on purpose to see if they would help. They barely did. I kept them in the final model anyway, and I report the comparison below.

### Do the extra features help?

| Features | Validation AUC | Test AUC |
|---|---:|---:|
| 3 (recency, orders, spent) | 0.7312 | 0.7300 |
| 5 (all of them) | 0.7335 | 0.7319 |

The gain is about 0.002, which is inside the noise (see the limitations section). The reason is that the new features mostly repeat information the model already had:

- `avg_order` is a direct calculation from two other features.
- `purchased_months` correlates 0.95 (Spearman) with `total_orders`.

Removing one feature at a time from the 5-feature model (validation AUC, full model = 0.7335):

| Removed | Validation AUC |
|---|---:|
| `recency` | 0.7261 |
| `total_orders` | 0.7322 |
| `total_spent` | 0.7318 |
| `avg_order` | 0.7313 |
| `purchased_months` | 0.7324 |

Only `recency` matters when it's removed. The others can replace each other.

## Models

I compared three models. All use validation AUC, and none of the trees needs scaled features. Logistic Regression is scaled inside a pipeline.

| Model | 3 features | 5 features |
|---|---:|---:|
| Logistic Regression | 0.7260 | 0.7276 |
| Random Forest (depth 6, 15 trees, min leaf 3) | 0.7291 | 0.7319 |
| HistGradientBoosting (100 iterations, depth 3, min leaf 81) | 0.7312 | 0.7335 |

I picked HistGradientBoosting because it was highest, but the differences between models are less than 0.01, so I wouldn't say it's clearly better. I tried a few values of `min_samples_leaf` and kept the best one on validation, so the validation number is a little optimistic.

### Threshold

The model gives a probability, and I need a cutoff to turn it into yes/no. I chose the cutoff that maximizes the F1 score for class 1 on the validation set: **0.2668**. The default 0.5 finds too few buyers. The F1 curve is flat around this point (validation F1 changes by about 0.002 between 0.25 and 0.27). The threshold was fixed before looking at the test set.

## Results on the test set (November)

AUC 0.7319, average precision 0.6330.

| | Predicted: no purchase | Predicted: purchase |
|---|---:|---:|
| Actually didn't buy | 1,981 | 652 |
| Actually bought | 551 | 790 |

| Metric | Value |
|---|---:|
| Accuracy | 69.7% |
| Precision (buyers) | 54.8% |
| Recall (buyers) | 58.9% |
| F1 (buyers) | 0.568 |
| F1 (non-buyers) | 0.767 |

The model catches about 59% of the people who bought, and about 55% of the people it flags actually buy.

### Compared with simple baselines

A model should beat something simple. Here, each baseline is one number used directly as a score, with no model:

| Score | Validation AUC | Test AUC |
|---|---:|---:|
| `recency` only (more recent = more likely) | 0.6555 | 0.6556 |
| `total_spent` only | 0.7083 | 0.7065 |
| `total_orders` only | 0.7154 | 0.7240 |
| **HistGradientBoosting, 5 features** | **0.7335** | **0.7319** |

The model beats "just count their orders", but only by about 0.01 AUC on validation and 0.008 on test. Most of the signal is in how often someone has bought before. Predicting "no purchase" for everyone gets 66.3% accuracy on November, so the 69.7% accuracy is only a small step up. The model's value is finding buyers, which that rule can't do.

### Ranking customers

Sorting November customers by predicted probability:

| Top of the list | Share who actually bought | Share of all buyers captured |
|---|---:|---:|
| Top 10% | 79.6% | 23.6% |
| Top 20% | 67.3% | 39.8% |
| Top 30% | 57.8% | 51.4% |

The overall rate was 33.7%, so the top 20% buys at about twice the average. If a campaign can only contact a small group, this is more useful than the accuracy number.

## Limitations

- **The scores move around between months.** On validation, the same model scores 0.741 AUC on September and 0.727 on October. Differences of 0.01 between models or feature sets are within that noise.
- **Only one test month**, and November is a pre-Christmas month where more people buy (33.7% vs 25.4% in training). One result doesn't show the model is stable over time. The accuracy, precision and F1 numbers also shift when the buying rate shifts.
- **One year of data**, so the model has never seen a full seasonal cycle.
- **Rows aren't independent.** The same customer appears in several snapshots, so the effective sample is smaller than the row count suggests.
- **The F1 threshold is generic.** A real campaign has a cost per contact and a value per sale, and the threshold should come from those.
- **The signal is moderate.** AUC around 0.73 with three or five basic features is a reasonable start, not a finished product.
- **The data is from 2010-2011**, so this demonstrates a workflow, not what to expect from a shop today.

## Repo layout

```
src/
  features.py   builds customer features and the target for a snapshot date
  modeling.py   FEATURES, THRESHOLD, create_model, train, predict
  train.py      runs the full pipeline and saves model.joblib
notebooks1/
  modeling.ipynb   exploration and model comparison
data/
  Online Retail.xlsx
requirements.txt
```

## How to run

Put the dataset at `data/Online Retail.xlsx`, then from the repo root:

```
pip install -r requirements.txt
python -m src.train
```

This builds the six snapshots, trains on June-August, prints the validation AUC (about 0.7335), and saves `model.joblib`.

To use the saved model:

```python
import joblib
from src.modeling import predict

model = joblib.load("model.joblib")
label, probability = predict(model, features_df)  # features_df from build_features()
```

`model.joblib` only loads reliably with the same scikit-learn version it was created with, so the versions in `requirements.txt` are pinned.

## Next steps

- Serve the model through an API (FastAPI) and put it in a Docker container.
- Choose the threshold from campaign costs instead of F1.
- Test on more months, and with rolling snapshots.
- Try other features, like the time between orders or spending trends.
