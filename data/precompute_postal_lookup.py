import pandas as pd
import pickle
import os

# Path to the Parquet file
PARQUET_PATH = "data/allCountries.parquet"
PICKLE_PATH = "data/postal_lookup.pkl"

if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(f"{PARQUET_PATH} not found.")

print("[INFO] Loading Parquet file...")
df = pd.read_parquet(PARQUET_PATH)

# Drop rows with missing key fields
df = df.dropna(subset=["postal_code", "place_name", "country_code"])

# Standardize column names and data types
text_columns = ["place_name", "state_name", "county_name", "postal_code", "country_code"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Normalize pincodes: remove spaces, uppercase
df['postal_code_norm'] = df['postal_code'].str.replace(' ', '').str.upper()

# Create lookup dictionary with normalized pincode
postal_lookup = {}
for _, row in df.iterrows():
    pincode = row['postal_code_norm']
    if pincode not in postal_lookup:
        postal_lookup[pincode] = []
    postal_lookup[pincode].append({
        'place_name': row['place_name'],
        'state_name': row['state_name'],
        'county_name': row['county_name'],
        'country_code': row['country_code'],
        'latitude': row['latitude'],
        'longitude': row['longitude']
    })

print(f"[INFO] Processed {len(postal_lookup)} unique normalized pincodes.")

# Save the lookup dictionary as a pickle file
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(postal_lookup, f)

print(f"[INFO] postal_lookup saved to {PICKLE_PATH}") 