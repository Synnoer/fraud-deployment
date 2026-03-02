from fastapi import FastAPI, UploadFile, File, Request
import onnxruntime as ort
import pandas as pd
import numpy as np
import joblib
import io
import time
from collections import deque

app = FastAPI()

# ---------------------------------------------------------------------------
# Startup — load model and unpack every artifact key
# ---------------------------------------------------------------------------
session   = ort.InferenceSession("W_fraud_model.onnx")
artifacts = joblib.load("preprocessing_artifacts.pkl")

label_encoders = artifacts['label_encoders']   # {col: fitted LabelEncoder}
scaler         = artifacts['scaler']           # fitted StandardScaler / MinMaxScaler
target_enc_map = artifacts['target_enc_map']   # {col: {value: target_mean_float}}
global_mean    = artifacts["global_mean"]
freq_encodings = artifacts['freq_encodings']   # {col: {value: frequency}}

static_cols      = artifacts['static_feature_cols']  # ordered — scaler input
seq_feature_cols = artifacts['seq_feature_cols']      # ordered — sequence buffer
cat_cols         = artifacts['cat_cols']              # ordered — embedding input
numeric_cols    = artifacts['numeric_cols']          # for sanity check before scaling
ordered_static_cols = artifacts['ordered_static_cols']

uid_feat_cols = artifacts['uid_feat_cols']   # per-user aggregate col names
time_features = artifacts['time_features']   # cyclic/calendar col names
agg_features  = artifacts['agg_features']    # rolling window col names
velocity_cols = artifacts['velocity_cols']   # time-delta / velocity col names
embed_dims    = artifacts['embed_dims']      # {col: dim} — informational only

fill_values = artifacts['fill_values']       # {col: training median/mode}

SEQ_LEN  = 15
TIME_COL = 'TransactionDT'
ID_COL   = 'TransactionID'

user_state: dict = {}


# ---------------------------------------------------------------------------
# Step 0 — hygiene (always before the main pipeline)
# ---------------------------------------------------------------------------

def step_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Inject any absent column using training fill values; fill NaNs in existing ones."""
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val
    return df


# ---------------------------------------------------------------------------
# Pipeline steps — order must match training exactly
# ---------------------------------------------------------------------------

def step_uid_features(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """
    Step 1 — Inject per-user aggregate features from warmup history.
    New users with no history fall back to artifact fill values.
    """
    uid_feats = state.get('uid_feats', {})
    for col in uid_feat_cols:
        df[col] = uid_feats.get(col, fill_values.get(col, 0))
    return df


def step_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 — Reproduce cyclic/calendar features from TransactionDT."""
    df['Transaction_hour'] = (df[TIME_COL] / 3600) % 24
    df['Transaction_day'] = (df[TIME_COL] / (3600 * 24)) % 7
    df['Transaction_week'] = (df[TIME_COL] / (3600 * 24 * 7))

    df['hour_sin'] = np.sin(2 * np.pi * df['Transaction_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Transaction_hour'] / 24)

    df['day_sin'] = np.sin(2 * np.pi * df['Transaction_day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['Transaction_day'] / 7)

    # Any additional time_features the artifact expects but weren't computed above
    for col in time_features:
        if col not in df.columns:
            df[col] = fill_values.get(col, 0)
    return df


def step_freq_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — Map categoricals to their training-set frequency values."""
    for col, freq_map in freq_encodings.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(-1)
    return df


def step_rolling_features(
    df: pd.DataFrame,
    state: dict,
    current_time: float,
    current_amt: float,
) -> pd.DataFrame:
    """
    Step 4 — Update 24-hour rolling window and write agg_features into df.
    Column names are taken from artifact['agg_features'] — not hardcoded.
    """
    state['rolling_window'].append((current_time, current_amt))
    state['rolling_window'] = [
        t for t in state['rolling_window'] if current_time - t[0] <= 86400
    ]
    window_amts = [t[1] for t in state['rolling_window']]
    cnt = len(window_amts)

    agg_map = {
        'roll_txn_count': cnt,
        'roll_amt_mean':  np.mean(window_amts) if cnt > 0 else -1,
        'roll_amt_std':   np.std(window_amts)  if cnt > 1 else 0.0,
        'roll_amt_max':   np.max(window_amts)  if cnt > 0 else -1,
        'roll_amt_min':   np.min(window_amts)  if cnt > 0 else -1,
    }
    for col in agg_features:
        df[col] = agg_map.get(col, fill_values.get(col, 0))

    return df


def step_velocity_features(
    df: pd.DataFrame,
    state: dict,
    current_time: float,
) -> pd.DataFrame:
    """
    Step 5 — Compute time-delta / velocity features and advance last_time in state.
    All column names come from artifact['velocity_cols'].
    """
    time_since_last = (current_time - state['last_time']) if state['last_time'] != -1 else -1
    state['last_time'] = current_time

    for col in velocity_cols:
        df[col] = time_since_last

    return df


def step_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6 — Label-encode cat_cols using fitted LabelEncoder objects from artifact."""
    for col in cat_cols:
        if col in df.columns:
            mapping = label_encoders[col]
            unseen_val = len(mapping)

            df[col] = (
                df[col]
                .astype(str)
                .map(mapping)
                .fillna(unseen_val)
                .astype(np.int32)
            )
    return df


def step_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7 — Apply target (mean) encoding using the training target_enc_map."""
    df["uid_target_enc"] = (
        df["uid"]
        .map(target_enc_map)
        .fillna(global_mean)
    )
    return df


def step_scale_static(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8 — Scale static features with the training scaler."""
    for col in static_cols:
        if col not in df.columns:
            df[col] = fill_values.get(col, 0)
            
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# ---------------------------------------------------------------------------
# Step 9 — sequence vector + ONNX input assembly
# ---------------------------------------------------------------------------

def step_build_seq_vector(df: pd.DataFrame) -> np.ndarray:
    """Step 9 — Extract per-event sequence vector aligned to seq_feature_cols."""
    for col in seq_feature_cols:
        if col not in df.columns:
            df[col] = fill_values.get(col, 0)
        else:
            df[col] = df[col].fillna(fill_values.get(col, 0))
    return df[seq_feature_cols].fillna(0).values.astype(np.float32)[0]


def validate_and_align(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """Guarantee exact column order and completeness before handing arrays to ONNX."""
    for col in col_list:
        if col not in df.columns:
            df[col] = fill_values.get(col, 0)
        else:
            df[col] = df[col].fillna(fill_values.get(col, 0))
    return df[col_list]


def build_onnx_inputs(df: pd.DataFrame, seq_buffer: list) -> dict:
    """Assemble the three tensors the CNN-GRU ONNX graph expects."""
    static_array = validate_and_align(df.copy(), static_cols).values.astype(np.float32)
    cat_array    = validate_and_align(df.copy(), cat_cols).values.astype(np.int64)[0]

    n_seq      = len(seq_feature_cols)
    seq_matrix = np.zeros((SEQ_LEN, n_seq), dtype=np.float32)
    buf_len    = min(len(seq_buffer), SEQ_LEN)
    if buf_len:
        seq_matrix[-buf_len:] = np.array(seq_buffer[-buf_len:])

    return {
        'input_static': static_array,
        'input_cat':    np.expand_dims(cat_array, axis=0),
        'input_seq':    np.expand_dims(seq_matrix, axis=0),
    }


# ---------------------------------------------------------------------------
# /warmup — replay history to build genuine per-user state
# ---------------------------------------------------------------------------

@app.post("/warmup")
async def warmup_state(file: UploadFile = File(...)):
    global user_state
    user_state.clear()

    contents = await file.read()
    df_all   = pd.read_csv(io.BytesIO(contents))
    df_all   = df_all.sort_values(TIME_COL).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # PHASE 1 — Stateless Batch Processing (Maximized Vectorization)
    # -----------------------------------------------------------------------
    df_all = step_fill_missing(df_all)
    df_all = step_time_features(df_all)
    df_all = step_freq_encoding(df_all) # Uses the updated batch-safe function
    
    # Batch Label Encoding
    for col in cat_cols:
        if col in df_all.columns:
            mapping = label_encoders[col]
            unseen_val = len(mapping)
            df_all[col] = df_all[col].astype(str).map(mapping).fillna(unseen_val).astype(np.int32)
        else:
            df_all[col] = 0

    df_all = step_target_encoding(df_all)

    # Pre-compute and batch-map UID features
    d1 = df_all['dist1'].fillna(-1) if 'dist1' in df_all.columns else 0
    d2 = df_all['dist2'].fillna(-1) if 'dist2' in df_all.columns else 0
    df_all['_uid_dist'] = d1 + d2

    uid_agg = (
        df_all.groupby('uid')
        .agg(
            uid_txn_count=('uid', 'count'),
            uid_amt_mean=('TransactionAmt', 'mean'),
            uid_amt_std=('TransactionAmt', 'std'),
            uid_dist_mean=('_uid_dist', 'mean'),
            uid_dist_std=('_uid_dist', 'std'),
            uid_email_nunique=('P_emaildomain', 'nunique'),
            uid_device_nunique=('DeviceInfo', 'nunique'),
        )
    )
    uid_agg['uid_amt_std']  = uid_agg['uid_amt_std'].fillna(-1)
    uid_agg['uid_dist_std'] = uid_agg['uid_dist_std'].fillna(-1)
    for col in uid_feat_cols:
        if col in uid_agg.columns:
            df_all[col] = df_all['uid'].map(uid_agg[col]).fillna(fill_values.get(col, 0))
        else:
            df_all[col] = fill_values.get(col, 0)

    # -----------------------------------------------------------------------
    # PHASE 2 — Stateful Processing (Chronological Loop)
    # -----------------------------------------------------------------------
    for idx, row in df_all.iterrows():
        uid = row.get('uid', row.get(ID_COL))

        if uid not in user_state:
            user_state[uid] = {
                'last_time':       -1,
                'rolling_window':  [],
                'sequence_buffer': [], # Will be populated in Phase 4
                'uid_feats':       uid_agg.loc[uid].to_dict() if uid in uid_agg.index else {},
            }

        state        = user_state[uid]
        current_time = float(row[TIME_COL])
        current_amt  = float(row['TransactionAmt'])

        # Calculate and record Rolling Features
        state['rolling_window'].append((current_time, current_amt))
        state['rolling_window'] = [t for t in state['rolling_window'] if current_time - t[0] <= 86400]
        window_amts = [t[1] for t in state['rolling_window']]
        cnt = len(window_amts)

        df_all.at[idx, 'roll_txn_count'] = cnt
        df_all.at[idx, 'roll_amt_mean']  = np.mean(window_amts) if cnt > 0 else -1
        df_all.at[idx, 'roll_amt_std']   = np.std(window_amts)  if cnt > 1 else 0.0
        df_all.at[idx, 'roll_amt_max']   = np.max(window_amts)  if cnt > 0 else -1
        df_all.at[idx, 'roll_amt_min']   = np.min(window_amts)  if cnt > 0 else -1

        # Calculate and record Velocity Features
        time_since_last = (current_time - state['last_time']) if state['last_time'] != -1 else -1
        state['last_time'] = current_time

        for col in velocity_cols:
            df_all.at[idx, col] = time_since_last

    # -----------------------------------------------------------------------
    # PHASE 3 — Batch Scaling
    # -----------------------------------------------------------------------
    # Ensure all static columns exist before passing to scaler
    for col in static_cols:
        if col not in df_all.columns:
            df_all[col] = fill_values.get(col, 0)
            
    df_all[numeric_cols] = scaler.transform(df_all[numeric_cols])

    # -----------------------------------------------------------------------
    # PHASE 4 — Sequence Buffer Construction
    # -----------------------------------------------------------------------
    # Ensure sequence columns exist
    for col in seq_feature_cols:
        if col not in df_all.columns:
            df_all[col] = fill_values.get(col, 0)

    # Extract the final SEQ_LEN scaled rows for each user
    for uid, group in df_all.groupby('uid'):
        seq_data = group[seq_feature_cols].values.astype(np.float32)
        # Convert the 2D array back into a list of 1D arrays to match stream format
        user_state[uid]['sequence_buffer'] = list(seq_data[-SEQ_LEN:])

    return {"status": "success", "users_warmed_up": len(user_state)}


# ---------------------------------------------------------------------------
# /predict_stream — single-event inference
# ---------------------------------------------------------------------------

@app.post("/predict_stream")
async def predict_stream(event: dict):
    """
    Accepts one raw transaction event and returns a fraud probability.
    Translates raw JSON into scaled numerical tensors while updating state.
    """
    uid = event.get('uid') or event.get(ID_COL)
    start_model = time.perf_counter()

    # Initialize a blank slate for a brand new user
    if uid not in user_state:
        user_state[uid] = {
            'last_time':       -1,
            'rolling_window':  [],
            'sequence_buffer': [],
            'uid_feats':       {}, # Will rely on fill_values in Step 1
        }
    state = user_state[uid]

    current_time = float(event.get(TIME_COL, 0))
    current_amt  = float(event.get('TransactionAmt', 0))

    # Isolate the event into a DataFrame
    df = pd.DataFrame([event])
    
    # CRITICAL: Ensure 'uid' exists in the dataframe so target_encoding (Step 7) doesn't fail
    df['uid'] = uid 

    # -----------------------------------------------------------------------
    # THE REAL-TIME TRANSLATION PIPELINE
    # -----------------------------------------------------------------------
    
    # Step 0: Inject missing columns with default fill values
    df = step_fill_missing(df)
    
    # Step 1: Inject historical UID aggregates (if any exist in state)
    df = step_uid_features(df, state)
    
    # Step 2: Translate raw timestamp into cyclic float features
    df = step_time_features(df)
    
    # Step 3: Translate categoricals to frequency frequencies (Safe version)
    df = step_freq_encoding(df)
    
    # Step 4: Update 24hr window in state and calculate new rolling aggregates
    df = step_rolling_features(df, state, current_time, current_amt)
    
    # Step 5: Compare new timestamp to state['last_time'] and overwrite it
    df = step_velocity_features(df, state, current_time)
    
    # Step 6: Translate categoricals to embedded integers (handles unseen)
    df = step_label_encoding(df)
    
    # Step 7: Apply target mean encoding based on UID
    df = step_target_encoding(df)
    
    # Step 8: Guarantee column alignment and scale static features
    df = step_scale_static(df)
    
    # -----------------------------------------------------------------------
    # SEQUENCE BUFFER & INFERENCE
    # -----------------------------------------------------------------------
    
    # Step 9: Extract the current event's sequence features and append to state
    seq_vec = step_build_seq_vector(df)
    state['sequence_buffer'].append(seq_vec)
    
    # Enforce the strict sliding window length
    if len(state['sequence_buffer']) > SEQ_LEN:
        state['sequence_buffer'].pop(0)

    # Step 10: Assemble final tensors and run the ONNX model
    input_data = build_onnx_inputs(df, state['sequence_buffer'])
    
    predictions = session.run(None, input_data)[0]
    model_latency = time.perf_counter() - start_model

    try:
        predictions = session.run(None, input_data)[0]
        # Handle different output shapes gracefully
        prob = float(
            predictions[0][0]
            if isinstance(predictions[0], (list, np.ndarray))
            else predictions[0]
        )
        if np.isnan(prob):
            prob = -2.0 
    except Exception as e:
        print(f"ONNX inference error for UID {uid}: {e}")
        prob = -1.0

    return {
        "uid":               uid,
        "TransactionDT":     current_time,
        "Fraud_Probability": prob,
        "model_latency_ms": round(model_latency * 1000, 3),
    }

request_times = deque(maxlen=1000)

@app.middleware("http")
async def track_rps(request: Request, call_next):
    request_times.append(time.time())
    response = await call_next(request)
    return response

def get_rps():
    now = time.time()
    return sum(t > now - 1 for t in request_times)