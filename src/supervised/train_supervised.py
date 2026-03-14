import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.supervised.feature_engine import AMLFeatureEngineer, HOLDOUT_ID

def run_supervised():
    os.makedirs("model_output", exist_ok=True)

    # 1. Train on Filtered
    eng_train = AMLFeatureEngineer("filtered_data")
    df_train = eng_train.prepare_dataset(include_labels=True)
    df_train = df_train[df_train['actual_label'].notna()]

    feats = ['structuring_count', 'wire_count', 'total_amount', 'high_risk_txn_count', 'age', 'tenure_years']
    X = df_train[feats]
    y = df_train['actual_label']

    pipe = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    pipe.fit(X, y)

    # 2. Predict on All Data
    eng_all = AMLFeatureEngineer("data")
    df_all = eng_all.prepare_dataset(include_labels=True) # Will treat holdout as NaN

    df_all['supervised_risk_score'] = pipe.predict_proba(df_all[feats])[:, 1]
    df_all['supervised_pred'] = (df_all['supervised_risk_score'] >= 0.5).astype(int)

    df_all[['customer_id', 'actual_label', 'supervised_risk_score', 'supervised_pred']].to_csv('model_output/supervised_preds.csv', index=False)

    # 3. Holdout Report
    print("\n" + "="*50)
    print("SUPERVISED HOLDOUT TEST")
    print("="*50)
    holdout_row = df_all[df_all['customer_id'] == HOLDOUT_ID]
    if not holdout_row.empty:
        score = holdout_row['supervised_risk_score'].values[0]
        print(f"Target ID: {HOLDOUT_ID}")
        print(f"Risk Score: {score:.1%} -> Flagged: {'YES' if score >= 0.5 else 'NO'}")
    else:
        print(f"Target ID {HOLDOUT_ID} not found in transactions.")

if __name__ == "__main__":
    run_supervised()