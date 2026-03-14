"""
Adapter that runs the moeez supervised model within the main pipeline.

Handles data path translation (data_o/ has KYC files nested under kyc/)
and returns supervised_risk_score as a DataFrame without writing to disk.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.supervised.feature_engine import AMLFeatureEngineer, HOLDOUT_ID
from src.config import DATA_RAW

log = logging.getLogger("aml_pipeline")

# KYC files live under data_o/kyc/, everything else is flat in data_o/
_KYC_FILES = {
    'kyc_individual.csv',
    'kyc_smallbusiness.csv',
    'kyc_small_business.csv',
    'kyc_industry_codes.csv',
    'kyc_occupation_codes.csv',
}


class PipelineFeatureEngineer(AMLFeatureEngineer):
    """Subclass that resolves flat filenames to actual data_o/ paths.

    The moeez AMLFeatureEngineer expects all CSVs in a single flat directory.
    In the main pipeline, KYC files are nested under data_o/kyc/.
    This subclass overrides _load_csv to handle that mapping.
    """

    def __init__(self):
        # Pass data_o/ as the base dir (used for non-KYC files)
        super().__init__(str(DATA_RAW))

    def _load_csv(self, filename):
        if filename in _KYC_FILES:
            path = os.path.join(str(DATA_RAW), "kyc", filename)
        else:
            path = os.path.join(str(DATA_RAW), filename)

        if os.path.exists(path):
            return pd.read_csv(path)
        return None


def run_supervised_in_pipeline():
    """Run the moeez supervised model and return scores as a DataFrame.

    Replicates the exact logic from moeez/train_supervised.py:
      1. Load all data via PipelineFeatureEngineer (resolves data_o/ paths)
      2. Train Random Forest + SMOTE on labeled data (holdout scrubbed by feature_engine)
      3. Predict on all customers
      4. Return DataFrame[customer_id, supervised_risk_score]

    Returns None if training fails (graceful fallback).
    """
    try:
        eng = PipelineFeatureEngineer()

        # Prepare dataset with labels (holdout is scrubbed from labels by feature_engine)
        df_all = eng.prepare_dataset(include_labels=True)
        if df_all is None:
            log.warning("Supervised model: no data returned from feature engineering")
            return None

        # Training set: only rows with known labels (holdout excluded by feature_engine)
        df_labeled = df_all[df_all['actual_label'].notna()].copy()
        if len(df_labeled) < 10:
            log.warning(f"Supervised model: only {len(df_labeled)} labeled samples, skipping")
            return None

        feats = ['structuring_count', 'wire_count', 'total_amount',
                 'high_risk_txn_count', 'age', 'tenure_years']
        X_train = df_labeled[feats]
        y_train = df_labeled['actual_label']

        # Exact same pipeline as moeez/train_supervised.py
        pipe = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('rf', RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ))
        ])
        pipe.fit(X_train, y_train)
        log.info(f"Supervised model trained on {len(df_labeled)} labeled samples "
                 f"({int(y_train.sum())} positive, {int(len(y_train) - y_train.sum())} negative)")

        # Predict on ALL customers (including holdout, who just has no training label)
        df_all['supervised_risk_score'] = pipe.predict_proba(df_all[feats])[:, 1]

        # Holdout report
        holdout_row = df_all[df_all['customer_id'] == HOLDOUT_ID]
        if not holdout_row.empty:
            score = holdout_row['supervised_risk_score'].values[0]
            log.info(f"Supervised holdout ({HOLDOUT_ID}): {score:.1%} -> "
                     f"{'FLAGGED' if score >= 0.5 else 'not flagged'}")

        n_flagged = (df_all['supervised_risk_score'] >= 0.5).sum()
        log.info(f"Supervised scores generated for {len(df_all):,} customers "
                 f"({n_flagged:,} flagged at >= 0.5)")

        return df_all[['customer_id', 'supervised_risk_score']].copy()

    except Exception as e:
        log.error(f"Supervised model failed: {e}", exc_info=True)
        return None