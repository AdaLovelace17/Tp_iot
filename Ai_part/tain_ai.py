"""
train_energy_model.py

Robust training pipeline for energy/power forecasting.
- Aggregates meter readings hourly per region (and global total).
- Merges environmental sensors (temperature, humidity) by hour & region.
- Builds lag and rolling features, cyclical time features.
- Time-based train/test split.
- Trains a RandomForestRegressor and saves model, scaler, metadata.

Usage: python train_energy_model.py
"""

"""
Simplified Energy Consumption Prediction Model
Suitable for small datasets (even < 1 hour of data)
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
import certifi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Connect to MongoDB
# ------------------------------
print("Connecting to MongoDB Atlas...")
mongo_client = MongoClient(
    "mongodb+srv://AdaLovelace:AdaLovelace1817@cluster0.jfdolkd.mongodb.net/?retryWrites=true&w=majority",
    tls=True,
    tlsCAFile=certifi.where()
)
db = mongo_client["SmartGrid"]
smartmeter_collection = db["smartmeters"]
env_collection = db["env_sensors"]

print(" Connected!")

# ------------------------------
# 2. Fetch and prepare data
# ------------------------------
meters_data = list(smartmeter_collection.find().sort("timestamp", -1).limit(5000))
env_data = list(env_collection.find().sort("timestamp", -1).limit(5000))

if len(meters_data) < 10:
    print("Not enough data. Collect more readings!")
    exit(1)

df_meters = pd.DataFrame(meters_data)

# Extract meter features
df_meters['voltage'] = df_meters['data'].apply(lambda x: x.get('voltage', 0) if isinstance(x, dict) else 0)
df_meters['current'] = df_meters['data'].apply(lambda x: x.get('current', 0) if isinstance(x, dict) else 0)
df_meters['power'] = df_meters['data'].apply(lambda x: x.get('power', 0) if isinstance(x, dict) else 0)

df_meters['timestamp'] = pd.to_datetime(df_meters['timestamp'], errors='coerce')
df_meters = df_meters.dropna(subset=['timestamp'])

# Time features
df_meters['hour'] = df_meters['timestamp'].dt.hour
df_meters['day_of_week'] = df_meters['timestamp'].dt.dayofweek
df_meters['month'] = df_meters['timestamp'].dt.month

# Region encoding
region_mapping = {'commercial':0, 'downtown':1, 'hybrid':2, 'port':3, 'residential':4, None:-1}
df_meters['region_encoded'] = df_meters['region_name'].map(region_mapping).fillna(-1)

# Environmental features
if len(env_data) > 0:
    df_env = pd.DataFrame(env_data)
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'], errors='coerce')
    df_env = df_env.dropna(subset=['timestamp'])
    df_env['hour'] = df_env['timestamp'].dt.hour

    temp_avg = df_env[df_env['sensor_type']=='temperature'].groupby('hour')['data'].mean()
    humidity_avg = df_env[df_env['sensor_type']=='humidity'].groupby('hour')['data'].mean()

    df_meters['temperature'] = df_meters['hour'].map(temp_avg).fillna(25)
    df_meters['humidity'] = df_meters['hour'].map(humidity_avg).fillna(60)
else:
    df_meters['temperature'] = 25
    df_meters['humidity'] = 60

# Features & target
features = ['hour', 'day_of_week', 'month', 'region_encoded', 'voltage', 'current', 'temperature', 'humidity']
X = df_meters[features].fillna(0)
y = df_meters['power']

# ------------------------------
# 3. Train model
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training R²: {train_score:.4f}, Testing R²: {test_score:.4f}")

# ------------------------------
# 4. Save model
# ------------------------------
joblib.dump(model, 'energy_model_simple.pkl')
joblib.dump(scaler, 'scaler_simple.pkl')
print("Model and scaler saved.")

# ------------------------------
# 5. Quick prediction test
# ------------------------------
sample_input = pd.DataFrame({
    'hour':[datetime.now().hour],
    'day_of_week':[datetime.now().weekday()],
    'month':[datetime.now().month],
    'region_encoded':[0],
    'voltage':[220],
    'current':[10],
    'temperature':[25],
    'humidity':[60]
})
sample_scaled = scaler.transform(sample_input)
prediction = model.predict(sample_scaled)[0]
print(f"Predicted Power for now: {prediction:.2f} kW")

mongo_client.close()

'''
import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import certifi
from pymongo import MongoClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
MONGO_URI = "mongodb+srv://AdaLovelace:AdaLovelace1817@cluster0.jfdolkd.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "SmartGrid"
SMART_COLLECTION = "smartmeters"
ENV_COLLECTION = "env_sensors"

# Output files
MODEL_FILE = "energy_prediction_model.pkl"
SCALER_FILE = "scaler.pkl"
METADATA_FILE = "model_metadata.json"
FEATURES_FILE = "model_features.json"

# Aggregation settings
RESAMPLE_FREQ = "H"  # hourly aggregation
MIN_RECORDS = 200  # minimum rows after aggregation to proceed
TEST_RATIO = 0.2  # last 20% time window as test

# Lag and rolling windows (in hours)
LAGS = [1, 3, 6, 24]
ROLL_WINDOWS = [3, 24]

# RandomForest params (tunable)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1
}

# -------------------------
# UTILITIES
# -------------------------
def safe_to_numeric(s):
    """Convert series to numeric coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")

def cyclical_features(df, col, period):
    """Add sin/cos features for cyclical variable (e.g., hour of day)."""
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df

# -------------------------
# 1) LOAD DATA FROM MONGO
# -------------------------
print("1) Connecting to MongoDB and fetching data...")
client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client[DB_NAME]
meters_col = db[SMART_COLLECTION]
env_col = db[ENV_COLLECTION]

# Fetch a reasonable number of records (most recent)
meters_docs = list(meters_col.find({}, projection=None).sort("timestamp", -1).limit(20000))
env_docs = list(env_col.find({}, projection=None).sort("timestamp", -1).limit(20000))

print(f"   - Fetched {len(meters_docs)} meter documents")
print(f"   - Fetched {len(env_docs)} env documents")

if len(meters_docs) == 0:
    raise SystemExit("No meter data found. Collect data before training.")

# -------------------------
# 2) BUILD DATAFRAMES
# -------------------------
print("2) Building DataFrames and normalizing fields...")

df_m = pd.DataFrame(meters_docs)
# Ensure timestamp exists and convert
df_m['timestamp'] = pd.to_datetime(df_m.get('timestamp', None), errors='coerce')
df_m = df_m.dropna(subset=['timestamp']).copy()

# Normalize nested 'data' dict: voltage,current,power,energy_consumed
def extract_field_from_data(df, field_name, default=0.0):
    return df['data'].apply(lambda x: x.get(field_name, default) if isinstance(x, dict) else default)

df_m['voltage'] = safe_to_numeric(extract_field_from_data(df_m, 'voltage', np.nan)).fillna(method='ffill').fillna(220)
df_m['current'] = safe_to_numeric(extract_field_from_data(df_m, 'current', np.nan)).fillna(0.0)
df_m['power'] = safe_to_numeric(extract_field_from_data(df_m, 'power', np.nan))
df_m['energy_consumed'] = safe_to_numeric(extract_field_from_data(df_m, 'energy_consumed', np.nan)).fillna(0.0)

# If power missing but voltage & current present, compute power
missing_power = df_m['power'].isna()
if missing_power.any():
    df_m.loc[missing_power, 'power'] = (df_m.loc[missing_power, 'voltage'] * df_m.loc[missing_power, 'current']).round(4)

# region_name cleanup (strip, lower)
df_m['region_name'] = df_m.get('region_name', pd.Series(index=df_m.index, dtype=object)).fillna('unknown').astype(str).str.strip().str.lower()

# env data
df_e = pd.DataFrame(env_docs)
if not df_e.empty:
    df_e['timestamp'] = pd.to_datetime(df_e.get('timestamp', None), errors='coerce')
    df_e = df_e.dropna(subset=['timestamp']).copy()
    df_e['region_name'] = df_e.get('region_name', pd.Series(index=df_e.index, dtype=object)).fillna('unknown').astype(str).str.strip().str.lower()
    df_e['sensor_type'] = df_e.get('sensor_type', pd.Series(index=df_e.index, dtype=object)).astype(str)
    df_e['value'] = safe_to_numeric(df_e.get('data', None)).fillna(np.nan)
else:
    # create empty df with expected columns
    df_e = pd.DataFrame(columns=['timestamp','region_name','sensor_type','value'])

print("   - Meter columns:", list(df_m.columns))
print("   - Env columns:", list(df_e.columns))

# -------------------------
# 3) AGGREGATE HOURLY (per region and global)
# -------------------------
print("3) Aggregating hourly metrics per region...")

# Choose whether to aggregate per region or global. We'll build per-region series then concatenate.
regions = df_m['region_name'].unique().tolist()
print("   - Found regions:", regions)

# We'll create a DataFrame 'agg_df' with MultiIndex (timestamp, region)
all_region_frames = []
for region in regions:
    sub = df_m[df_m['region_name'] == region].copy()
    if sub.empty:
        continue
    # set index and resample hourly: compute mean voltage/current/power, count samples
    sub = sub.set_index('timestamp').sort_index()
    hourly = sub[['voltage','current','power','energy_consumed']].resample(RESAMPLE_FREQ).agg({
        'voltage':'mean',
        'current':'mean',
        'power':'mean',
        'energy_consumed':'sum'
    }).rename(columns={
        'voltage':'voltage_mean',
        'current':'current_mean',
        'power':'power_mean',
        'energy_consumed':'energy_sum'
    })
    hourly['region_name'] = region
    all_region_frames.append(hourly)

if not all_region_frames:
    raise SystemExit("No hourly aggregated frames produced. Check incoming data.")

agg_df = pd.concat(all_region_frames).reset_index()
# If you also want a global aggregation add:
global_hourly = agg_df.groupby('timestamp').agg({
    'voltage_mean':'mean',
    'current_mean':'mean',
    'power_mean':'mean',
    'energy_sum':'sum'
}).reset_index().assign(region_name='__global__')

agg_df = pd.concat([agg_df, global_hourly], ignore_index=True).sort_values(['timestamp','region_name']).reset_index(drop=True)

print("   - Aggregated rows (hour x region):", len(agg_df))

# -------------------------
# 4) MERGE ENVIRONMENTAL (temperature, humidity) per hour+region
# -------------------------
print("4) Merging environmental features (hourly averages per region)...")
# pivot env sensors to hourly avg by region & sensor_type
if not df_e.empty:
    df_e = df_e.set_index('timestamp').sort_index()
    env_hourly = df_e.groupby(['region_name', pd.Grouper(freq=RESAMPLE_FREQ), 'sensor_type'])['value'].mean().reset_index()
    # pivot sensor types to columns
    env_pivot = env_hourly.pivot_table(index=['timestamp','region_name'], columns='sensor_type', values='value')
    env_pivot = env_pivot.reset_index().rename_axis(None, axis=1)
    # merge
    agg_df = agg_df.merge(env_pivot, how='left', left_on=['timestamp','region_name'], right_on=['timestamp','region_name'])
else:
    # fill with defaults columns if missing
    agg_df['temperature'] = np.nan
    agg_df['humidity'] = np.nan

# Fill defaults for env if missing
agg_df['temperature'] = agg_df.get('temperature', np.nan).fillna(25.0)
agg_df['humidity'] = agg_df.get('humidity', np.nan).fillna(60.0)

# -------------------------
# 5) CLEANING & OUTLIER HANDLING
# -------------------------
print("5) Cleaning and outlier handling...")

# drop rows where power_mean is nan
before = len(agg_df)
agg_df = agg_df.dropna(subset=['power_mean'])
after = len(agg_df)
print(f"   - dropped {before-after} rows with no power")

# remove unrealistic voltage/current/power
agg_df = agg_df[(agg_df['voltage_mean'].between(180, 260)) & (agg_df['current_mean'] >= 0)]
# clamp extreme power using quantiles
p_low, p_high = agg_df['power_mean'].quantile([0.01, 0.99])
agg_df['power_mean'] = agg_df['power_mean'].clip(lower=p_low, upper=p_high)

# -------------------------
# 6) FEATURE ENGINEERING (lags, rolling, cyclical, encode)
# -------------------------
print("6) Feature engineering: lags, rolling, cyclical time features, encoding regions...")

# Sort properly
agg_df = agg_df.sort_values(['region_name','timestamp']).reset_index(drop=True)

# Add time features
agg_df['hour'] = agg_df['timestamp'].dt.hour
agg_df['day_of_week'] = agg_df['timestamp'].dt.dayofweek
agg_df['month'] = agg_df['timestamp'].dt.month
agg_df['year'] = agg_df['timestamp'].dt.year

# cyclical encoding for hour and day_of_week
agg_df = cyclical_features(agg_df, 'hour', 24)
agg_df = cyclical_features(agg_df, 'day_of_week', 7)

# region one-hot (keep global '__global__' too)
agg_df['region_name'] = agg_df['region_name'].fillna('unknown')
region_dummies = pd.get_dummies(agg_df['region_name'], prefix='region')
agg_df = pd.concat([agg_df, region_dummies], axis=1)

# For lag features and rolling, operate per region
for lag in LAGS:
    agg_df[f'lag_power_{lag}'] = agg_df.groupby('region_name')['power_mean'].shift(lag)

for w in ROLL_WINDOWS:
    agg_df[f'roll_mean_{w}'] = agg_df.groupby('region_name')['power_mean'].shift(1).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)

# EMAs
agg_df['ema_3'] = agg_df.groupby('region_name')['power_mean'].shift(1).ewm(span=3, adjust=False).mean().reset_index(level=0, drop=True)
agg_df['ema_24'] = agg_df.groupby('region_name')['power_mean'].shift(1).ewm(span=24, adjust=False).mean().reset_index(level=0, drop=True)

# Fill remaining NaNs after lags with forward/backfill sensible defaults
agg_df = agg_df.sort_values('timestamp')
agg_df[['lag_power_1','lag_power_3','lag_power_6','lag_power_24','roll_mean_3','roll_mean_24','ema_3','ema_24']] = agg_df[['lag_power_1','lag_power_3','lag_power_6','lag_power_24','roll_mean_3','roll_mean_24','ema_3','ema_24']].fillna(method='ffill').fillna(method='bfill')

# Final feature selection
feature_cols = [
    # time
    'hour','day_of_week','month',
    'hour_sin','hour_cos','day_of_week_sin','day_of_week_cos',
    # measurements
    'voltage_mean','current_mean',
    # env
    'temperature','humidity',
    # historical
    'lag_power_1','lag_power_3','lag_power_6','lag_power_24',
    'roll_mean_3','roll_mean_24','ema_3','ema_24'
]

# add region dummies columns
region_cols = [c for c in agg_df.columns if c.startswith('region_')]
feature_cols += region_cols

# Check we have features present
missing_feats = [c for c in feature_cols if c not in agg_df.columns]
if missing_feats:
    raise SystemExit(f"Missing expected features: {missing_feats}")

# define X,y (predict next-hour power for same region)
# target column: power_mean shifted - predict 1-step ahead
agg_df['target_power_next'] = agg_df.groupby('region_name')['power_mean'].shift(-1)

# drop rows where target is NaN (last hour per region)
agg_df = agg_df.dropna(subset=['target_power_next'])
print("   - total samples after feature engineering:", len(agg_df))

if len(agg_df) < MIN_RECORDS:
    raise SystemExit(f"Not enough aggregated samples ({len(agg_df)}). Need >= {MIN_RECORDS} after aggregation.")

X = agg_df[feature_cols].astype(float).fillna(0.0)
y = agg_df['target_power_next'].astype(float)

# -------------------------
# 7) TIME-BASED SPLIT (train/test)
# -------------------------
print("7) Time-based train/test split...")

# sort by timestamp to preserve time order
agg_df = agg_df.sort_values('timestamp').reset_index(drop=True)
split_index = int((1 - TEST_RATIO) * len(agg_df))
train_idx = agg_df.index[:split_index]
test_idx = agg_df.index[split_index:]

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"   - train samples: {len(X_train)}, test samples: {len(X_test)}")

# -------------------------
# 8) SCALE & TRAIN
# -------------------------
print("8) Scaling features and training model...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_train_scaled, y_train)

# -------------------------
# 9) EVALUATE
# -------------------------
print("9) Evaluating model...")

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"   - Train R2: {train_r2:.4f}")
print(f"   - Test  R2: {test_r2:.4f}")
print(f"   - Test  MAE: {test_mae:.3f}")
print(f"   - Test  RMSE: {test_rmse:.3f}")

fi = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\n Top feature importances:")
print(fi.head(20).to_string(index=False))

# -------------------------
# 10) SAVE MODEL & ARTIFACTS
# -------------------------
print("\n10) Saving model, scaler, metadata...")

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

metadata = {
    'trained_date': datetime.utcnow().isoformat(),
    'n_samples': len(X),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'test_mae': float(test_mae),
    'test_rmse': float(test_rmse),
    'features': feature_cols,
    'lags': LAGS,
    'rolling': ROLL_WINDOWS,
    'resample_freq': RESAMPLE_FREQ,
    'rf_params': RF_PARAMS
}

with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=2)

with open(FEATURES_FILE, 'w') as f:
    json.dump(feature_cols, f, indent=2)

print(f"   - Model saved to: {MODEL_FILE}")
print(f"   - Scaler saved to: {SCALER_FILE}")
print(f"   - Metadata saved to: {METADATA_FILE}")

# -------------------------
# 11) QUICK MULTI-HOUR FORECAST (iterative) demo
# -------------------------
print("\n11) Multi-hour iterative forecast demo (next 6 hours)")

# pick a recent sample (last available timestamp for a chosen region, here choose global)
sample_region = '__global__'
recent = agg_df[agg_df['region_name'] == sample_region].sort_values('timestamp').tail(1)
if recent.empty:
    # fallback to first region found
    sample_region = regions[0]
    recent = agg_df[agg_df['region_name'] == sample_region].sort_values('timestamp').tail(1)

if recent.empty:
    print("No recent sample found for forecast demo.")
else:
    prev_row = recent.iloc[-1:].copy()
    forecasts = []
    row = prev_row.copy()
    for h in range(1, 7):
        # build feature vector for next hour: update hour/day/month and lag features
        next_time = row['timestamp'].iloc[0] + pd.Timedelta(hours=1)
        next_hour = next_time.hour
        next_dow = next_time.dayofweek if hasattr(next_time, 'dayofweek') else pd.Timestamp(next_time).dayofweek
        # create a new row dict based on prev but shifted
        new_row = row.copy().reset_index(drop=True)
        new_row.at[0,'timestamp'] = next_time
        new_row.at[0,'hour'] = next_hour
        new_row.at[0,'day_of_week'] = next_dow
        new_row = cyclical_features(new_row, 'hour', 24)
        new_row = cyclical_features(new_row, 'day_of_week', 7)
        # shift lag features
        new_row.at[0,'lag_power_24'] = new_row.at[0,'lag_power_24']  # keep older
        # compose feature vector
        X_next = new_row[feature_cols].astype(float).fillna(0.0)
        X_next_scaled = scaler.transform(X_next)
        pred = model.predict(X_next_scaled)[0]
        forecasts.append((next_time, float(pred)))
        # now push forward: update lags rolling etc (simple iterative approach)
        # create row for next iter: power_mean becomes predicted
        row = new_row.copy()
        row.at[0,'power_mean'] = pred
        # update lag features: shift manually (simplified)
        for lag in sorted(LAGS, reverse=True):
            key = f'lag_power_{lag}'
            if lag == 1:
                row.at[0, 'lag_power_1'] = pred
            else:
                # keep from previous (this is an approximation)
                row.at[0, key] = row.get(key, np.nan)
    print(" Forecasts (next 6 hours):")
    for t,p in forecasts:
        print(f"  {t.strftime('%Y-%m-%d %H:%M')} -> {p:.2f} kW")

print("\nTraining pipeline finished.")
client.close()
'''