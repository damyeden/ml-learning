"""
Zindi Financial Health Prediction Challenge - Advanced Solution >= 0.91 Target
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import os

# =============================================================================
# 1. LOAD DATA
# =============================================================================
DATA_DIR = r'c:\Users\Eden\ml-learning\m4'
train = pd.read_csv(os.path.join(DATA_DIR, 'Train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'SampleSubmission.csv'))

target_col = 'Target'
id_col = 'ID'

target_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_train = train[target_col].map(target_map).values.astype(int)

train_ids = train[id_col].values
test_ids = test[id_col].values
train.drop(columns=[id_col, target_col], inplace=True)
test.drop(columns=[id_col], inplace=True)

combined = pd.concat([train, test], axis=0, ignore_index=True)
n_train = len(train)

# =============================================================================
# 2. FEATURE ENGINEERING & CATEGORICAL PREP
# =============================================================================
numeric_cols = ['owner_age', 'personal_income', 'business_expenses', 
                'business_turnover', 'business_age_years', 'business_age_months']
categorical_cols = [c for c in combined.columns if c not in numeric_cols]

# Clean categorical text
for col in categorical_cols:
    combined[col] = combined[col].astype(str).str.strip().str.lower()
    combined[col] = combined[col].replace({
        'nan': 'missing',
        '': 'missing',
        'refused': 'missing'
    })
    # Group all "don't know" variants
    for val in combined[col].unique():
        if isinstance(val, str) and any(x in val for x in ["don't know", "don?t know", "do not know", "n/a", "doesn?t apply"]):
            combined[col] = combined[col].replace({val: "dont_know"})

# --- Create engineered features ---
def create_features(df):
    # Financial ratios
    df['turnover_to_expenses'] = df['business_turnover'] / (df['business_expenses'] + 1)
    df['income_to_expenses'] = df['personal_income'] / (df['business_expenses'] + 1)
    df['income_to_turnover'] = df['personal_income'] / (df['business_turnover'] + 1)
    df['profit_proxy'] = df['business_turnover'] - df['business_expenses']
    df['profit_margin'] = df['profit_proxy'] / (df['business_turnover'] + 1)
    df['expense_ratio'] = df['business_expenses'] / (df['business_turnover'] + 1)
    
    # Business maturity
    df['total_business_age_months'] = df['business_age_years'].fillna(0) * 12 + df['business_age_months'].fillna(0)
    df['turnover_per_year'] = df['business_turnover'] / (df['business_age_years'] + 1)
    df['income_per_age'] = df['personal_income'] / (df['owner_age'] + 1)
    
    # Log transforms for skewed financial features
    for col in ['personal_income', 'business_expenses', 'business_turnover']:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0).fillna(0))
    df['profit_proxy_log'] = np.log1p(df['profit_proxy'].clip(lower=0).fillna(0))
    
    # Count Encodings for ALL categorical features (Frequency of each category)
    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq)
        
    # Insurance count
    insurance_cols = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance', 'has_insurance']
    for c in insurance_cols:
        df[f'{c}_have_now'] = (df[c] == 'have now').astype(int)
    df['insurance_count'] = sum(df[f'{c}_have_now'] for c in insurance_cols)
    
    # Financial product count
    fin_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account', 'has_internet_banking', 'has_debit_card']
    for c in fin_cols:
        df[f'{c}_have_now'] = (df[c] == 'have now').astype(int)
    df['financial_products_count'] = sum(df[f'{c}_have_now'] for c in fin_cols)
    
    # Attitude score
    att_cols = ['attitude_stable_business_environment', 'attitude_satisfied_with_achievement', 'attitude_more_successful_next_year']
    for c in att_cols:
        df[f'{c}_yes'] = (df[c] == 'yes').astype(int)
    df['positive_attitude_score'] = sum(df[f'{c}_yes'] for c in att_cols)
    
    # Risk score
    df['worried_shutdown'] = (df['attitude_worried_shutdown'] == 'yes').astype(int)
    df['cash_flow_problem'] = (df['current_problem_cash_flow'] == 'yes').astype(int)
    df['sourcing_money_problem'] = (df['problem_sourcing_money'] == 'yes').astype(int)
    df['risk_score'] = df['worried_shutdown'] + df['cash_flow_problem'] + df['sourcing_money_problem']
    
    # Formality score
    df['income_tax_compliant'] = (df['compliance_income_tax'] == 'yes').astype(int)
    df['keeps_records_yes'] = df['keeps_financial_records'].astype(str).str.contains('yes', na=False).astype(int)
    df['formality_score'] = df['income_tax_compliant'] + df['keeps_records_yes']
    
    # Interactions
    df['products_x_turnover_log'] = df['financial_products_count'] * df['business_turnover_log']
    df['insurance_x_income_log'] = df['insurance_count'] * df['personal_income_log']
    df['attitude_x_products'] = df['positive_attitude_score'] * df['financial_products_count']
    df['formality_x_products'] = df['formality_score'] * df['financial_products_count']
    
    return df

combined = create_features(combined)

# Set categorical columns as formal 'category' dtype for LightGBM/XGBoost native support
for col in categorical_cols:
    # Ensure absolutely no NaNs are left as CatBoost will fail
    combined[col] = combined[col].fillna('missing').astype(str)
    # Convert to pandas category dtype
    combined[col] = combined[col].astype('category')

X_full = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

print(f"Features: {X_full.shape[1]}")

# =============================================================================
# 3. ADVANCED MODEL TRAINING (NATIVE CATEGORICAL)
# =============================================================================
N_SPLITS = 10
N_SEEDS = 3
SEEDS = [42, 2024, 777]

# We will ensemble LGBM, XGB, CAT using native categorical features
def get_lgbm_model(seed):
    return lgb.LGBMClassifier(
        objective='multiclass', num_class=3, boosting_type='gbdt',
        learning_rate=0.015, num_leaves=63, max_depth=7,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5, n_estimators=3000,
        random_state=seed, class_weight='balanced', verbose=-1
    )

def get_xgb_model(seed):
    return xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        learning_rate=0.015, max_depth=6, min_child_weight=7,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
        n_estimators=3000, random_state=seed, enable_categorical=True, 
        tree_method='hist', verbosity=0
    )

def get_cat_model(seed):
    return CatBoostClassifier(
        loss_function='MultiClass', learning_rate=0.02, depth=6,
        iterations=3000, l2_leaf_reg=5, random_seed=seed, verbose=0,
        auto_class_weights='Balanced', bootstrap_type='Bernoulli', subsample=0.8,
        cat_features=categorical_cols
    )

oof_lgb = np.zeros((n_train, 3))
oof_xgb = np.zeros((n_train, 3))
oof_cat = np.zeros((n_train, 3))

test_lgb = np.zeros((len(test), 3))
test_xgb = np.zeros((len(test), 3))
test_cat = np.zeros((len(test), 3))

print("=" * 60)
print("STARTING OOF TRAINING")
print("=" * 60)

for seed in SEEDS:
    print(f"\n--- SEED {seed} ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_train)):
        X_tr, y_tr = X_full.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_full.iloc[va_idx], y_train[va_idx]
        
        # --- LightGBM ---
        m_lgb = get_lgbm_model(seed)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_lgb[va_idx] += m_lgb.predict_proba(X_va) / N_SEEDS
        test_lgb += m_lgb.predict_proba(X_test) / (N_SPLITS * N_SEEDS)
        
        # --- XGBoost ---
        m_xgb = get_xgb_model(seed)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] += m_xgb.predict_proba(X_va) / N_SEEDS
        test_xgb += m_xgb.predict_proba(X_test) / (N_SPLITS * N_SEEDS)
        
        # --- CatBoost ---
        m_cat = get_cat_model(seed)
        # CatBoost native Pool
        train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols)
        valid_pool = Pool(X_va, y_va, cat_features=categorical_cols)
        
        m_cat.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150, verbose_eval=False)
        oof_cat[va_idx] += m_cat.predict_proba(X_va) / N_SEEDS
        test_cat += m_cat.predict_proba(X_test) / (N_SPLITS * N_SEEDS)
        
        print(f"Fold {fold} complete.")

# Calculate individual OOF F1 scores
print("\n" + "="*60)
print("INDIVIDUAL OOF SCORES")
print("="*60)
f1_lgb = f1_score(y_train, np.argmax(oof_lgb, axis=1), average='weighted')
f1_xgb = f1_score(y_train, np.argmax(oof_xgb, axis=1), average='weighted')
f1_cat = f1_score(y_train, np.argmax(oof_cat, axis=1), average='weighted')

print(f"LightGBM OOF F1: {f1_lgb:.6f}")
print(f"XGBoost OOF F1:  {f1_xgb:.6f}")
print(f"CatBoost OOF F1: {f1_cat:.6f}")

# =============================================================================
# 4. ENSEMBLE OPTIMIZATION
# =============================================================================
print("\n" + "="*60)
print("ENSEMBLE OPTIMIZATION")
print("="*60)

best_f1 = 0
best_w = None

steps = np.arange(0.0, 1.05, 0.05)
for w_lgb in steps:
    for w_xgb in steps:
        w_cat = round(1.0 - w_lgb - w_xgb, 2)
        if w_cat < 0 or w_cat > 1.0:
            continue
            
        blend = (w_lgb * oof_lgb) + (w_xgb * oof_xgb) + (w_cat * oof_cat)
        f1 = f1_score(y_train, np.argmax(blend, axis=1), average='weighted')
        
        if f1 > best_f1:
            best_f1 = f1
            best_w = (w_lgb, w_xgb, w_cat)

print(f"Best Weights (LGB, XGB, CAT): {best_w}")
print(f"Best Ensemble OOF F1: {best_f1:.6f}")

# Final prediction
final_blend = (best_w[0] * test_lgb) + (best_w[1] * test_xgb) + (best_w[2] * test_cat)
final_preds = np.argmax(final_blend, axis=1)

# =============================================================================
# 5. SUBMISSION
# =============================================================================
reverse_map = {0: 'Low', 1: 'Medium', 2: 'High'}
final_labels = [reverse_map[c] for c in final_preds]

submission = sample_sub.copy()
submission['Target'] = final_labels

print(f"\nSubmission distribution:\n{submission['Target'].value_counts()}")

sub_path = os.path.join(DATA_DIR, 'submission_native_cat.csv')
submission.to_csv(sub_path, index=False)
print(f"\n✅ Advanced Native Cat Submission saved to: {sub_path}")
print(f"   Submit this file to Zindi to crack the 0.91 barrier!")
