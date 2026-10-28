from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["recency", "total_orders", "total_spent", "avg_order", "purchased_months"]
THRESHOLD = 0.266791 ## chose it on validation by max F1 for this feature set

def create_model():
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        min_samples_leaf=81, 
        max_depth=3,
        early_stopping=False,
        random_state=42
    )
    return model


def train(final_df):

    model = create_model().fit(final_df[FEATURES], final_df['target'])
    return model


def predict(model, features_df):

    prob = model.predict_proba(features_df[FEATURES])[:, 1]
    label = (prob >= THRESHOLD).astype(int)

    return label, prob