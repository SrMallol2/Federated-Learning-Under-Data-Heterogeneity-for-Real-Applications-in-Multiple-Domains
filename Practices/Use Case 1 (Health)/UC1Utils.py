"""
Here everything related to data loading, cleaning, and feature engineering 
for the diabetes readmission dataset lives. 
The same functions are used by both the centralized and federated notebooks, 
with some additional federated-specific helpers at the bottom.

This stored here and not in the notebooks to avoid cluttering the main 
workflow (Createing a FL model and compare it with the centralized version) 
with long data processing code.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import os
import zipfile

COLS_TO_DROP = ['weight', 'max_glu_serum', 'A1Cresult', 
                'medical_specialty', 'payer_code', 
                'encounter_id']

DECEASED_IDS = [11, 13, 14, 19, 20, 21]

ADMISSION_TYPE_MAP = {
    1: 'emergency', 2: 'urgent',   3: 'elective',
    4: 'newborn',   5: 'unknown',  6: 'unknown',
    7: 'emergency', 8: 'unknown'
}

DISCHARGE_DISPOSITION_MAP = {
    11: 'expired', 19: 'expired', 20: 'expired', 21: 'expired',
    1: 'home', 6: 'home', 8: 'home',
    2: 'transfer', 3: 'transfer', 4: 'transfer', 5: 'transfer',
    10: 'transfer', 15: 'transfer', 16: 'transfer', 22: 'transfer',
    23: 'transfer', 24: 'transfer', 27: 'transfer', 28: 'transfer',
    29: 'transfer', 30: 'transfer',
    13: 'hospice', 14: 'hospice',
    7: 'ama',
    9: 'inpatient', 12: 'inpatient',
    18: 'unknown', 25: 'unknown', 26: 'unknown',
}

ADMISSION_SOURCE_MAP = {
    1: 'referral', 2: 'referral', 3: 'referral',
    4: 'transfer', 5: 'transfer', 6: 'transfer',
    10: 'transfer', 18: 'transfer', 22: 'transfer', 
    25: 'transfer', 26: 'transfer',
    7: 'emergency',
    11: 'newborn', 12: 'newborn', 13: 'newborn',
    14: 'newborn', 23: 'newborn', 24: 'newborn',
    9: 'unknown', 15: 'unknown', 17: 'unknown',
    20: 'unknown', 21: 'unknown',
    8: 'other', 19: 'other',
}

AGE_MAP = {
    '[0-10)': 0,  '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
    '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
    '[80-90)': 8, '[90-100)': 9
}

MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

MED_MAP = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

DATA_DIR   = "../diabetes_data"                          
CSV_MAIN   = os.path.join(DATA_DIR, "diabetic_data.csv")
CSV_IDS    = os.path.join(DATA_DIR, "IDS_mapping.csv")
ZIP_FILE   = os.path.join(DATA_DIR, "diabetes.zip")   # adjust name to match your zip

def ensure_data():
    csvs_present = os.path.exists(CSV_MAIN) and os.path.exists(CSV_IDS)

    if csvs_present:
        print("✓ CSV files already present, skipping extraction.")
        return

    if os.path.exists(ZIP_FILE):
        print(f"Extracting {ZIP_FILE} …")
        with zipfile.ZipFile(ZIP_FILE, "r") as zf:
            zf.extractall(DATA_DIR)
        print("✓ Extraction complete.")
    else:
        raise FileNotFoundError(
            f"Neither the CSV files nor '{ZIP_FILE}' were found in '{DATA_DIR}'.\n"
            "Please add the zip file to the repo or place the CSVs manually."
        )


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and replace ? with NaN."""
    df = pd.read_csv(path)
    df.replace('?', np.nan, inplace=True)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missing and ID columns."""
    return df.drop(columns=COLS_TO_DROP)


def remove_deceased(df: pd.DataFrame) -> pd.DataFrame:
    """Remove patients who died — they cannot be readmitted."""
    return df[~df['discharge_disposition_id'].isin(DECEASED_IDS)].copy()


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target: 1 if readmitted within 30 days, 0 otherwise."""
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    df.drop(columns=['readmitted'], inplace=True)
    return df


def group_discharge(x):
    if pd.isna(x):
        return 'unknown'
    return DISCHARGE_DISPOSITION_MAP.get(x, 'other')


def group_icd9(code):
    if pd.isna(code):
        return 'other'
    code = str(code).strip()
    if code.startswith('V') or code.startswith('E'):
        return 'other'
    try:
        c = float(code)
    except ValueError:
        return 'other'
    if   390 <= c <= 459 or c == 785: return 'circulatory'
    elif 460 <= c <= 519 or c == 786: return 'respiratory'
    elif 520 <= c <= 579 or c == 787: return 'digestive'
    elif 250 <= c <= 250.99:          return 'diabetes'
    elif 800 <= c <= 999:             return 'injury'
    elif 710 <= c <= 739:             return 'musculoskeletal'
    elif 580 <= c <= 629 or c == 788: return 'genitourinary'
    elif 140 <= c <= 239:             return 'neoplasms'
    else:                             return 'other'


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all encoding and grouping."""
    # Ordinal
    df['age'] = df['age'].map(AGE_MAP)

    # Clinical groupings
    df['admission_type_id'] = df['admission_type_id'].map(ADMISSION_TYPE_MAP).fillna('unknown')
    df['admission_source_id']     = df['admission_source_id'].map(ADMISSION_SOURCE_MAP).fillna('unknown')
    df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(group_discharge)

    # ICD-9 diagnosis codes
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(group_icd9)

    # Medication columns
    for col in MED_COLS:
        if col in df.columns:
            df[col] = df[col].map(MED_MAP)

    df['change']      = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=[
        'diag_1', 'diag_2', 'diag_3',
        'admission_type_id', 'admission_source_id',
        'discharge_disposition_id', 'race', 'gender'
    ])

    return df


def prepare_data(path: str, verbose: bool = True) -> tuple:
    df = load_data(path)
    df = drop_columns(df)
    df = remove_deceased(df)
    df = create_target(df)

    # Impute race — affects ~2% of rows
    df['race'] = df['race'].fillna('Unknown')
    # diag NaN handled inside group_icd9 (returns 'other') — no fillna needed

    # Fix #4: log rows dropped by dropna so silent data loss is visible
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if verbose and dropped > 0:
        print(f"dropna removed {dropped} rows ({dropped/before*100:.2f}%)"
              f" — likely from unmapped admission_type/source IDs")

    df = encode_features(df)

    if verbose:
        print(f"Dataset shape after cleaning: {df.shape}")
        print(f"Class distribution:\n{df['readmitted_binary'].value_counts()}")

    # Fix #3: capture feature names here before converting to numpy
    feature_names = df.drop(columns=['readmitted_binary', 'patient_nbr']).columns.tolist()

    groups = df['patient_nbr'].values
    X = df.drop(columns=['readmitted_binary', 'patient_nbr']).values
    y = df['readmitted_binary'].values

    return X, y, groups, feature_names



# ── Federated-only helpers ────────────────────────────────────────────────────
# Not used by the centralized notebook. Added here to avoid a second utils file.
 
def prepare_data_aligned(path: str, global_columns: list, verbose: bool = True) -> tuple:
    """
    Same as prepare_data but aligns the encoded DataFrame to a pre-defined
    column schema. Required for federated clients: each client runs get_dummies
    independently on its own data slice, which may be missing rare dummy columns
    (e.g. a client with no 'newborn' admissions). Without alignment the feature
    dimensions would differ across clients, making model aggregation impossible.
 
    Parameters
    ----------
    path           : path to a client's raw_data.csv
    global_columns : feature name list from derive_global_columns()
    verbose        : print progress info
 
    Returns
    -------
    X, y, groups, feature_names  — same signature as prepare_data()
    """
    df = load_data(path)
    df = drop_columns(df)
    df = remove_deceased(df)
    df = create_target(df)
 
    df['race'] = df['race'].fillna('Unknown')
 
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if verbose and dropped > 0:
        print(f"dropna removed {dropped} rows ({dropped/before*100:.2f}%)")
 
    df = encode_features(df)
 
    # Add any columns present in the global schema but absent in this client's
    # data (rare categories not seen locally), filled with 0.
    for col in global_columns:
        if col not in df.columns:
            df[col] = 0
 
    # Drop any extra columns and enforce global column order.
    meta = [c for c in ['patient_nbr', 'readmitted_binary'] if c in df.columns]
    df   = df[meta + global_columns]
 
    if verbose:
        print(f"Dataset shape after cleaning: {df.shape}")
        print(f"Class distribution:\n{df['readmitted_binary'].value_counts()}")
 
    groups = df['patient_nbr'].values
    X      = df[global_columns].values.astype(np.float32)
    y      = df['readmitted_binary'].values.astype(np.int64)
 
    return X, y, groups, global_columns
 
 
def derive_global_columns(data_path: str) -> list:
    """
    Run prepare_data on the full dataset to learn the complete dummy column
    space produced by get_dummies. Returns the canonical feature name list
    to pass as global_columns to every prepare_data_aligned call.
    """
    print("Deriving global feature schema from full dataset...")
    _, _, _, feature_names = prepare_data(data_path, verbose=False)
    print(f"  {len(feature_names)} feature columns")
    return feature_names
 
 
def split_data(
    X:         np.ndarray,
    y:         np.ndarray,
    groups:    np.ndarray,
    test_size: float = 0.2,
    val_size:  float = 0.15,
    seed:      int   = 42,
) -> tuple:
    """
    Patient-level train / val / test split mirroring the centralized baseline.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tv_idx, test_idx = next(gss.split(X, y, groups=groups))
 
    X_tv, X_test = X[tv_idx],    X[test_idx]
    y_tv, y_test = y[tv_idx],    y[test_idx]
    g_tv         = groups[tv_idx]
 
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(X_tv, y_tv, groups=g_tv))
 
    return (
        X_tv[train_idx], X_tv[val_idx], X_test,
        y_tv[train_idx], y_tv[val_idx], y_test,
    )
 
 
def scale_data(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    Fit StandardScaler on train only, transform val and test.
    Returns X_train_scaled, X_val_scaled, X_test_scaled, scaler.
    """
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler