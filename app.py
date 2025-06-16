from flask import Flask, render_template, request, jsonify
import pandas as pd
from functools import lru_cache
import os
from rapidfuzz import fuzz
import math

app = Flask(__name__)

@lru_cache(maxsize=1)
def load_all_postal_data():
    path = "data/allCountries.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError("data/allCountries.parquet not found.")

    df = pd.read_parquet(path)
    df = df.dropna(subset=["postal_code", "place_name", "country_code"])

    text_columns = ["place_name", "state_name", "county_name", "postal_code", "country_code"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df

def parse_latlon(raw):
    try:
        lat, lon = map(float, raw.strip().split(","))
        return lat, lon
    except Exception as e:
        print(f"[DEBUG] Failed to parse latlon '{raw}': {e}")
        return None, None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * (2 * math.asin(math.sqrt(a)))

def find_matching_locations(area, pincode, raw_latlon, df, name_threshold=70, geo_threshold_km=20.0):
    area = area.strip().lower()
    pincode = pincode.strip().lower()
    lat, lon = parse_latlon(raw_latlon)

    candidates = df[df["postal_code"] == pincode]
    if candidates.empty:
        return False, []

    matched_rows = []

    for _, row in candidates.iterrows():
        name_match = any(
            pd.notna(row[col]) and fuzz.token_sort_ratio(area, row[col]) >= name_threshold
            for col in ["place_name", "state_name", "county_name"]
        )

        if not name_match:
            continue

        # Enforce haversine only if lat/lon is supplied
        if lat is not None and lon is not None:
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                continue
            dist_km = haversine(float(row["latitude"]), float(row["longitude"]), lat, lon)
            if dist_km > geo_threshold_km:
                continue  # Skip if far away

        matched_rows.append({
            "place_name": row["place_name"],
            "state_name": row["state_name"],
            "county_name": row["county_name"],
            "postal_code": row["postal_code"],
            "latitude": str(row["latitude"]),
            "longitude": str(row["longitude"])
        })
    print(f"[DEBUG] Area={area}, Pincode={pincode}, LatLon=({lat}, {lon}), Matches={len(matched_rows)}")

    return bool(matched_rows), matched_rows

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/validate", methods=["POST"])
def validate():
    try:
        area = request.form["area"]
        pincode = request.form["pincode"]
        latlon = request.form.get("latlon", "").strip()
        print(f"[DEBUG] Raw latlon input: '{latlon}'")  # Format: "12.932402, 77.608947"
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    df = load_all_postal_data()
    is_valid, matches = find_matching_locations(area, pincode, latlon, df)

    return jsonify({"valid": is_valid, "matches": matches if is_valid else []})

if __name__ == "__main__":
    app.run(debug=True)
