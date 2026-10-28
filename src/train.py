from src.modeling import build_features, build_target
from src.features import train, predict
from sklearn.metrics import roc_auc_score
import joblib
import pandas as pd

SNAPSHOTS = ["2011-06-01", "2011-07-01", "2011-08-01", 
             "2011-09-01", "2011-10-01", "2011-11-01"]

def main():
    
    df = pd.read_excel("data/Online Retail.xlsx")
    df['revenue'] = df['Quantity'] * df['UnitPrice']
    parts = []

    for date in SNAPSHOTS:
        
        features = build_features(df, date)
        target = build_target(df, date)  
        dataset = features.join(target, how='inner')

        dataset["snapshot_date"] = pd.to_datetime(date)
        parts.append(dataset)
    
    all_df = pd.concat(parts)

    train_df = all_df[all_df['snapshot_date'] < '2011-09-01']  
    val_df = all_df[(all_df['snapshot_date'] >= '2011-09-01') & 
                    (all_df['snapshot_date'] < '2011-10-01')]      
    
    
    model = train(train_df)
    
    label, prob = predict(model, val_df)
    print(roc_auc_score(val_df["target"], prob))    

    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    main()