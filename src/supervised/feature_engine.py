import pandas as pd
import numpy as np
import os

HOLDOUT_ID = 'SYNID0107832828'

class AMLFeatureEngineer:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _load_csv(self, filename):
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path) if os.path.exists(path) else None

    def prepare_dataset(self, include_labels=False):
        # 1. Load KYC
        kyc_ind = self._load_csv('kyc_individual.csv')
        kyc_biz = self._load_csv('kyc_smallbusiness.csv')
        if kyc_biz is None: kyc_biz = self._load_csv('kyc_small_business.csv')

        if kyc_ind is not None:
            kyc_ind = kyc_ind[['customer_id', 'occupation_code', 'birth_date', 'onboard_date', 'income']]
            kyc_ind['type'] = 'Individual'
        if kyc_biz is not None:
            kyc_biz = kyc_biz[['customer_id', 'industry_code', 'established_date', 'onboard_date', 'sales']]
            kyc_biz.rename(columns={'sales': 'income'}, inplace=True)
            kyc_biz['type'] = 'Business'

        kyc_all = pd.concat([d for d in [kyc_ind, kyc_biz] if d is not None], ignore_index=True)

        # 2. Load Transactions
        tx_files = {
            'card.csv': 'card', 'abm.csv': 'abm', 'eft.csv': 'eft',
            'emt.csv': 'emt', 'wire.csv': 'wire',
            'westernunion.csv': 'westernunion', 'cheque.csv': 'cheque'
        }
        all_txns = []
        for file, ch in tx_files.items():
            df = self._load_csv(file)
            if df is not None:
                df['channel'] = ch
                all_txns.append(df)
        if not all_txns: return None
        tx = pd.concat(all_txns, ignore_index=True)

        # 3. Aggregate Features
        tx['amount_cad'] = pd.to_numeric(tx['amount_cad'], errors='coerce').fillna(0)

        feats = tx.groupby('customer_id').agg(
            txn_count=('transaction_id', 'count'),
            total_amount=('amount_cad', 'sum'),
            std_amount=('amount_cad', 'std')
        ).reset_index()

        tx['is_structuring'] = (tx['amount_cad'] >= 9000) & (tx['amount_cad'] < 10000)
        tx['is_wire'] = tx['channel'] == 'wire'
        tx['is_high_risk'] = tx['channel'].isin(['westernunion', 'wire', 'abm'])

        risk = tx.groupby('customer_id').agg(
            structuring_count=('is_structuring', 'sum'),
            wire_count=('is_wire', 'sum'),
            high_risk_txn_count=('is_high_risk', 'sum')
        ).reset_index()

        feats = feats.merge(risk, on='customer_id', how='left').merge(kyc_all, on='customer_id', how='left')

        now = pd.Timestamp.now()
        if 'birth_date' in feats.columns:
            feats['birth_date'] = pd.to_datetime(feats['birth_date'], errors='coerce')
            feats['age'] = (now - feats['birth_date']).dt.days / 365.25
        if 'onboard_date' in feats.columns:
            feats['onboard_date'] = pd.to_datetime(feats['onboard_date'], errors='coerce')
            feats['tenure_years'] = (now - feats['onboard_date']).dt.days / 365.25

        # 4. Attach Labels & Apply Holdout
        if include_labels:
            labels = self._load_csv('labels.csv')
            if labels is not None:
                # SCRUB THE HOLDOUT ID SO IT BECOMES UNLABELLED
                labels = labels[labels['customer_id'] != HOLDOUT_ID]
                labels = labels.rename(columns={'label': 'actual_label'})
                feats = feats.merge(labels, on='customer_id', how='left')

        return feats