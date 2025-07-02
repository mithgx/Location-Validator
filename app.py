from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
from functools import lru_cache
import os
from rapidfuzz import fuzz
import math
import tempfile
import json
import io
import numpy as np
import gc
import hashlib
from datetime import datetime
import pickle

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Constants for optimized batch processing
MAX_DISPLAY_ROWS = 20

@lru_cache(maxsize=1)
def load_all_postal_data():
    """Load and cache postal data, preferring the pickle file for speed."""
    pickle_path = "data/postal_lookup.pkl"
    parquet_path = "data/allCountries.parquet"
    if os.path.exists(pickle_path):
        print(f"[INFO] Loading postal lookup from pickle: {pickle_path}")
        with open(pickle_path, "rb") as f:
            postal_lookup = pickle.load(f)
        print(f"[INFO] Loaded {len(postal_lookup)} unique normalized pincodes from pickle.")
        return postal_lookup
    else:
        print("[WARNING] Pickle file not found, loading from Parquet (this will be slow)...")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"{parquet_path} not found.")
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=["postal_code", "place_name", "country_code"])
        text_columns = ["place_name", "state_name", "county_name", "postal_code", "country_code"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df['postal_code_norm'] = df['postal_code'].str.replace(' ', '').str.upper()
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
        print(f"[INFO] Loaded {len(postal_lookup)} unique normalized pincodes from Parquet.")
        return postal_lookup

def normalize_input_data(df, column_mapping):
    """Normalize and clean input data"""
    normalized_df = df.copy()
    # Standardize column names
    for col in ['area', 'city', 'state', 'country', 'latitude', 'longitude', 'pincode', 'latlong']:
        if column_mapping.get(col) in normalized_df.columns:
            normalized_df[col] = normalized_df[column_mapping[col]]
    # Clean and normalize data
    for col in ['area', 'city', 'state', 'country']:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].astype(str).str.lower().str.strip()
    if 'pincode' in normalized_df.columns:
        normalized_df['pincode'] = normalized_df['pincode'].astype(str).str.upper().str.strip()
    if 'latlong' in normalized_df.columns:
        normalized_df['latlong'] = normalized_df['latlong'].astype(str).str.strip()
    # Handle missing values
    for col in ['area', 'city', 'state', 'country', 'pincode', 'latlong']:
        if col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].replace(['nan', 'none', ''], '')
    return normalized_df

def normalize_pincode(pincode):
    """Normalize pincode by removing spaces and making uppercase"""
    if not pincode or pd.isna(pincode):
        return ''
    return str(pincode).replace(' ', '').upper()

def parse_latlon(latlon_str):
    """Parse a latlon string like '12.93, 77.60' into floats. Returns (lat, lon) or (None, None) if invalid."""
    try:
        if not latlon_str or not isinstance(latlon_str, str):
            return None, None
        parts = latlon_str.split(',')
        if len(parts) != 2:
            return None, None
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except Exception:
        return None, None

def is_within_threshold(lat1, lon1, lat2, lon2, threshold=0.1):
    """Check if two lat/lon pairs are within the threshold (in degrees)."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return False
    return abs(lat1 - lat2) <= threshold and abs(lon1 - lon2) <= threshold

def load_country_name_to_code_map():
    mapping = {}
    path = "data/countryInfo.txt"
    if not os.path.exists(path):
        print("[WARNING] countryInfo.txt not found. Country name mapping will not work.")
        return mapping
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) > 4:
                code = parts[0].strip().upper()
                name = parts[4].strip().lower()
                mapping[name] = code
    return mapping

COUNTRY_NAME_TO_CODE = load_country_name_to_code_map()

def normalize_country_to_code(country_input):
    if not country_input:
        return ''
    country_input = country_input.strip().lower()
    # Try mapping from countryInfo.txt
    code = COUNTRY_NAME_TO_CODE.get(country_input)
    if code:
        return code
    # Common fallbacks
    fallback_map = {
        'usa': 'US',
        'us': 'US',
        'united states': 'US',
        'united states of america': 'US',
        'india': 'IN',
        'in': 'IN',
        'bharat': 'IN',
        'australia': 'AU',
        'au': 'AU',
        # Add more as needed
    }
    if country_input in fallback_map:
        return fallback_map[country_input]
    # fallback: if already a 2-letter code, return as uppercase
    if len(country_input) == 2:
        return country_input.upper()
    return ''

def fast_validate_batch(df, postal_lookup):
    """Ultra-fast batch validation using vectorized operations, with optional lat/lon check."""
    try:
        results = []
        df['formatted_pincode'] = df['pincode'].apply(normalize_pincode)
        valid_mask = (df['formatted_pincode'] != '') & ((df.get('area', '') != '') | (df.get('city', '') != ''))
        valid_df = df[valid_mask].copy()
        print(f"[DEBUG] Processing {len(valid_df)} valid rows out of {len(df)} total")
        batch_size = 1000
        for i in range(0, len(valid_df), batch_size):
            batch = valid_df.iloc[i:i+batch_size]
            batch_results = []
            for _, row in batch.iterrows():
                area = row.get('area', '').lower().strip() if 'area' in row else ''
                city = row.get('city', '').lower().strip() if 'city' in row else ''
                state = row.get('state', '').lower().strip() if 'state' in row else ''
                country = row.get('country', '').lower().strip() if 'country' in row else ''
                country_code = normalize_country_to_code(row.get('country', ''))
                formatted_pincode = row['formatted_pincode']
                lat_input, lon_input = parse_latlon(row.get('latlong', '')) if row.get('latlong', '') else (None, None)
                is_valid = False
                matched_area = ''
                matched_city = ''
                matched_state = ''
                matched_country = ''
                matched_latitude = ''
                matched_longitude = ''
                matched_pincode = ''
                debug_failed = False
                threshold = 80 if country_code == 'IN' else 70
                if formatted_pincode in postal_lookup:
                    postal_records = postal_lookup[formatted_pincode]
                    for postal_record in postal_records:
                        pr_area = postal_record['place_name'].lower().strip()
                        pr_state = postal_record['state_name'].lower().strip()
                        pr_county = postal_record['county_name'].lower().strip()
                        pr_country = postal_record['country_code'].upper().strip()
                        record_fields = [pr_area, pr_state, pr_county]
                        # If city matches, mark as valid immediately
                        city_match = False
                        if city:
                            for record_val in record_fields:
                                if (
                                    fuzz.token_sort_ratio(city, record_val) >= threshold or
                                    fuzz.partial_ratio(city, record_val) >= threshold or
                                    record_val in city or city in record_val
                                ):
                                    city_match = True
                                    break
                        if city_match:
                            # Check country match before marking as valid
                            normalized_input_country = normalize_country_to_code(row.get('country', ''))
                            normalized_reference_country = normalize_country_to_code(postal_record.get('country_code', ''))
                            if not normalized_input_country or normalized_input_country == normalized_reference_country:
                                is_valid = True
                                matched_area = postal_record['place_name']
                                matched_city = postal_record['place_name']
                                matched_state = postal_record['state_name']
                                matched_country = postal_record['country_code']
                                matched_latitude = postal_record.get('latitude', '')
                                matched_longitude = postal_record.get('longitude', '')
                                matched_pincode = formatted_pincode
                                break
                            else:
                                # Country mismatch, skip this record
                                continue
                        # Cross-field fuzzy/partial/substring match for area (must match independently)
                        area_match = False
                        if area:
                            for record_val in record_fields:
                                if (
                                    fuzz.token_sort_ratio(area, record_val) >= threshold or
                                    fuzz.partial_ratio(area, record_val) >= threshold or
                                    record_val in area or area in record_val
                                ):
                                    area_match = True
                                    break
                        else:
                            area_match = True
                        if not area_match:
                            debug_failed = True
                            continue
                        # Cross-field fuzzy/partial/substring match for city (must match independently)
                        # (already checked above, so skip here)
                        # Cross-field fuzzy/partial/substring match for state
                        state_match = False
                        if state:
                            for record_val in record_fields:
                                if (
                                    fuzz.token_sort_ratio(state, record_val) >= threshold or
                                    fuzz.partial_ratio(state, record_val) >= threshold or
                                    record_val in state or state in record_val
                                ):
                                    state_match = True
                                    break
                        else:
                            state_match = True
                        if not state_match:
                            debug_failed = True
                            continue
                        # Country code strict match (must match if provided)
                        country_match = True
                        # Always normalize input country to code and compare to uppercase reference
                        normalized_input_country = normalize_country_to_code(row.get('country', ''))
                        normalized_reference_country = normalize_country_to_code(postal_record.get('country_code', ''))
                        if normalized_input_country:
                            country_match = (normalized_input_country == normalized_reference_country)
                        if not country_match:
                            print(f"[DEBUG] Country mismatch: input='{row.get('country', '')}', normalized_input='{normalized_input_country}', reference='{postal_record.get('country_code', '')}', normalized_reference='{normalized_reference_country}'")
                            debug_failed = True
                            continue
                        # Lat/lon match (if provided)
                        latlon_match = True
                        if lat_input is not None and lon_input is not None:
                            latlon_match = is_within_threshold(lat_input, lon_input, postal_record.get('latitude'), postal_record.get('longitude'))
                        if latlon_match:
                            is_valid = True
                            matched_area = postal_record['place_name']
                            matched_city = postal_record['place_name']
                            matched_state = postal_record['state_name']
                            matched_country = postal_record['country_code']
                            matched_latitude = postal_record.get('latitude', '')
                            matched_longitude = postal_record.get('longitude', '')
                            matched_pincode = formatted_pincode
                            break
                    if not is_valid and debug_failed:
                        print(f"[DEBUG] No match for input: area='{area}', city='{city}', state='{state}', country='{country_code}', pincode='{formatted_pincode}'")
                        print(f"[DEBUG] Postal records for pincode {formatted_pincode}:")
                        for rec in postal_records:
                            print(rec)
                # Compute difference column
                difference_fields = []
                def norm_str(val):
                    return str(val).strip().lower() if val is not None else ''
                def norm_num(val):
                    try:
                        return float(val)
                    except:
                        return str(val).strip().lower() if val is not None else ''
                def floats_close(a, b, tol=1):
                    try:
                        return abs(float(a) - float(b)) <= tol
                    except:
                        return str(a).strip().lower() == str(b).strip().lower()
                if norm_str(row.get('city', '')) != norm_str(matched_city):
                    difference_fields.append('City')
                if norm_str(row.get('state', '')) != norm_str(matched_state):
                    difference_fields.append('State')
                input_country_code = normalize_country_to_code(row.get('country', ''))
                matched_country_code = normalize_country_to_code(matched_country)
                if input_country_code != matched_country_code:
                    difference_fields.append('Country')
                if not floats_close(row.get('latitude', ''), matched_latitude):
                    difference_fields.append('Latitude')
                if not floats_close(row.get('longitude', ''), matched_longitude):
                    difference_fields.append('Longitude')
                if norm_str(row.get('pincode', '')) != norm_str(matched_pincode):
                    difference_fields.append('Pincode')
                difference = ', '.join(difference_fields)
                batch_results.append({
                    "input_area": row.get('area', ''),
                    "input_city": row.get('city', ''),
                    "input_state": row.get('state', ''),
                    "input_country": row.get('country', ''),
                    "input_latitude": row.get('latitude', '') if 'latitude' in row else (lat_input if lat_input is not None else ''),
                    "input_longitude": row.get('longitude', '') if 'longitude' in row else (lon_input if lon_input is not None else ''),
                    "input_pincode": row.get('pincode', ''),
                    "matched_area": matched_area,
                    "matched_city": matched_city,
                    "matched_state": matched_state,
                    "matched_country": matched_country,
                    "matched_latitude": matched_latitude,
                    "matched_longitude": matched_longitude,
                    "matched_pincode": matched_pincode,
                    "valid": is_valid,
                    "difference": difference
                })
            results.extend(batch_results)
            if (i + batch_size) % 10000 == 0:
                print(f"[DEBUG] Processed {i + batch_size}/{len(valid_df)} rows")
        invalid_df = df[~valid_mask]
        for _, row in invalid_df.iterrows():
            lat_input, lon_input = parse_latlon(row.get('latlong', '')) if row.get('latlong', '') else (None, None)
            latitude = lat_input if lat_input is not None else ''
            longitude = lon_input if lon_input is not None else ''
            country_code = normalize_country_to_code(row.get('country', ''))
            results.append({
                "input_area": row.get('area', ''),
                "input_city": row.get('city', ''),
                "input_state": row.get('state', ''),
                "input_country": row.get('country', ''),
                "input_latitude": latitude,
                "input_longitude": longitude,
                "input_pincode": row.get('pincode', ''),
                "matched_area": '',
                "matched_city": '',
                "matched_state": '',
                "matched_country": country_code,
                "matched_latitude": latitude,
                "matched_longitude": longitude,
                "matched_pincode": '',
                "valid": False,
                "difference": ''
            })
        return results
    except Exception as e:
        print(f"[ERROR] Error in fast_validate_batch: {str(e)}")
        return []

def find_matching_locations(area, pincode, latlon, postal_lookup):
    """Find matching locations for single validation, with optional lat/lon check."""
    try:
        formatted_pincode = normalize_pincode(pincode)
        if not formatted_pincode:
            print(f"[DEBUG] Invalid pincode format: {pincode}")
            return False, []
        matches = []
        lat_input, lon_input = parse_latlon(latlon) if latlon else (None, None)
        if formatted_pincode in postal_lookup:
            postal_records = postal_lookup[formatted_pincode]
            area = area.lower().strip()
            for record in postal_records:
                area_match = (
                    area in record['place_name'] or 
                    area in record['state_name'] or 
                    area in record['county_name']
                )
                country_code = normalize_country_to_code(record.get('country_code', ''))
                latlon_match = True
                if lat_input is not None and lon_input is not None:
                    latlon_match = is_within_threshold(lat_input, lon_input, record.get('latitude'), record.get('longitude'))
                if area_match and latlon_match:
                    matches.append({
                        'city': record['place_name'],
                        'state': record['state_name'],
                        'country': record.get('county_name', ''),
                        'country_code': country_code,
                        'pincode': formatted_pincode,
                        'latitude': record.get('latitude', 0.0),
                        'longitude': record.get('longitude', 0.0)
                    })
        is_valid = len(matches) > 0
        return is_valid, matches
    except Exception as e:
        print(f"[ERROR] Error in find_matching_locations: {str(e)}")
        return False, []

def find_locations_by_area(area_or_pincode, postal_lookup):
    """Find locations by area name or pincode, returning all available info for each match."""
    try:
        query = area_or_pincode.lower().strip()
        matches = []
        # Search through all postal records
        for prefix, records in postal_lookup.items():
            # Match by pincode (exact or partial)
            if query in prefix.lower():
                for record in records:
                    match = dict(record)
                    match['postal_code'] = prefix
                    matches.append(match)
            # Match by area/city/state/county
            for record in records:
                if (
                    query in record.get('place_name', '').lower() or
                    query in record.get('state_name', '').lower() or
                    query in record.get('county_name', '').lower()
                ):
                    match = dict(record)
                    match['postal_code'] = prefix
                    matches.append(match)
        return matches
    except Exception as e:
        print(f"[ERROR] Error in find_locations_by_area: {str(e)}")
        return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/validate", methods=["POST"])
def validate():
    try:
        area = request.form["area"]
        pincode = request.form["pincode"]
        latlon = request.form.get("latlon", "").strip()
        print(f"[DEBUG] Raw latlon input: '{latlon}'")
        print(f"[DEBUG] Raw pincode input: '{pincode}'")
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    postal_lookup = load_all_postal_data()
    is_valid, matches = find_matching_locations(area, pincode, latlon, postal_lookup)

    return jsonify({"valid": is_valid, "matches": matches if is_valid else []})

@app.route("/search", methods=["POST"])
def search():
    try:
        area_or_pincode = request.form["area"]
        if not area_or_pincode.strip():
            return jsonify({"error": "Area name or pincode is required"}), 400
        postal_lookup = load_all_postal_data()
        matches = find_locations_by_area(area_or_pincode, postal_lookup)
        return jsonify({"matches": matches})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        return jsonify({"error": "An error occurred while searching for locations"}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload with optimized processing"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.xlsx'):
            return jsonify({"error": "Please upload an Excel (.xlsx) file"}), 400
        
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        file.save(temp_file.name)
        
        # Read Excel file to get column names and preview
        try:
            df = pd.read_excel(temp_file.name, nrows=5)  # Read first 5 rows for column names
            columns = df.columns.tolist()
            preview = df.head(5).to_dict(orient='records')  # Get preview data
            total_rows = len(pd.read_excel(temp_file.name))  # Get total rows
        except Exception as e:
            os.unlink(temp_file.name)
            return jsonify({"error": f"Failed to read Excel file: {str(e)}"}), 400
        
        # Store file path in session
        session['uploaded_file'] = temp_file.name
        
        return jsonify({
            "success": True,
            "columns": columns,
            "preview": preview,
            "total_rows": total_rows,
            "temp_file": temp_file.name,
            "message": "File uploaded successfully. Please map the columns."
        })
        
    except Exception as e:
        print(f"[ERROR] File upload failed: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/validate-batch", methods=["POST"])
def validate_batch():
    """Ultra-fast batch validation"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        temp_file = data.get('tempFile') or session.get('uploaded_file')
        if not temp_file or not os.path.exists(temp_file):
            return jsonify({"error": "Invalid or missing file data"}), 400

        column_mapping = data.get('columnMapping', {})
        if not all(key in column_mapping for key in ['area', 'pincode', 'latlong']):
            return jsonify({"error": "Missing required column mappings"}), 400

        print("[DEBUG] Starting ultra-fast batch validation...")
        
        # Load postal data with optimized lookup
        postal_lookup = load_all_postal_data()
        print(f"[DEBUG] Loaded postal lookup with {len(postal_lookup)} prefixes")
        
        # Read and normalize input data
        print("[DEBUG] Reading and normalizing input data...")
        df = pd.read_excel(temp_file)
        normalized_df = normalize_input_data(df, column_mapping)
        total_rows = len(normalized_df)
        
        print(f"[DEBUG] Processing {total_rows} rows with ultra-fast validation")
        
        # Process all data at once with optimized logic
        start_time = datetime.now()
        all_results = fast_validate_batch(normalized_df, postal_lookup)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        total_processed = len(all_results)
        print(f"[DEBUG] Successfully processed {total_processed} rows in {processing_time:.2f} seconds")
        
        # Limit results for display only, keep all for download
        display_results = all_results[:MAX_DISPLAY_ROWS]
        
        # Store results in temporary file instead of session
        results_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        
        # Write results in chunks to avoid memory issues
        with open(results_file.name, 'w') as f:
            json.dump(all_results, f, separators=(',', ':'))  # Compact JSON format
        
        # Store file path in session (small data)
        session['results_file'] = results_file.name
        session['total_processed'] = total_processed
        
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except Exception as e:
            print(f"[WARNING] Failed to delete temporary file: {str(e)}")

        # Force garbage collection immediately
        del normalized_df, df, all_results
        gc.collect()

        return jsonify({
            "results": display_results,
            "total_processed": total_processed,
            "display_count": len(display_results),
            "download_count": total_processed,
            "processing_time": processing_time
        })

    except Exception as e:
        print(f"[ERROR] Batch validation failed: {str(e)}")
        return jsonify({"error": f"Failed to validate the data: {str(e)}"}), 500

@app.route("/download-results", methods=["POST"])
def download_results():
    """Download validation results with optimized streaming"""
    try:
        # Get results file path from session
        results_file = session.get('results_file')
        if not results_file or not os.path.exists(results_file):
            return jsonify({"error": "No results available for download"}), 400

        print(f"[DEBUG] Downloading from results file: {results_file}")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'validation_results_{timestamp}.xlsx'
        
        # Create Excel file directly from JSON with optimization
        output = io.BytesIO()
        
        # Read JSON in chunks for memory efficiency
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        if not results_data:
            return jsonify({"error": "No results available for download"}), 400

        print(f"[DEBUG] Loaded {len(results_data)} results for download")

        # Convert to DataFrame efficiently
        df = pd.DataFrame(results_data)
        # Ensure all required columns are present and in order
        required_columns = [
            'input_area', 'input_city', 'input_state', 'input_country', 'input_latitude', 'input_longitude', 'input_pincode',
            'matched_area', 'matched_city', 'matched_state', 'matched_country', 'matched_latitude', 'matched_longitude', 'matched_pincode', 'valid', 'difference'
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        df = df[required_columns]
        df.columns = [col.upper() for col in df.columns]
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.upper()
        
        # Use xlsxwriter for better performance
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Capitalize column names for the Excel output
            df.columns = [col.upper() for col in df.columns]
            df.to_excel(writer, index=False, sheet_name='Validation Results')
            workbook = writer.book
            worksheet = writer.sheets['Validation Results']
            header_format = workbook.add_format({
                'bold': True,
                'border': 1
            })
            cell_format = workbook.add_format({'border': 1})
            # Overwrite header row with custom format (all uppercase, no blue background)
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 15)
            if len(df) > 0:
                worksheet.conditional_format(1, 0, len(df), len(df.columns) - 1, {'type': 'no_blanks', 'format': cell_format})
                worksheet.conditional_format(1, 0, len(df), len(df.columns) - 1, {'type': 'blanks', 'format': cell_format})
            if 'VALID' in df.columns:
                valid_col_idx = df.columns.get_loc('VALID')
                nrows = len(df) + 1  # +1 for header
                worksheet.conditional_format(1, valid_col_idx, nrows, valid_col_idx, {
                    'type': 'cell',
                    'criteria': '==',
                    'value': True,
                    'format': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                })
                worksheet.conditional_format(1, valid_col_idx, nrows, valid_col_idx, {
                    'type': 'cell',
                    'criteria': '==',
                    'value': False,
                    'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                })

        output.seek(0)
        
        # Check if output has data
        output_size = len(output.getvalue())
        print(f"[DEBUG] Generated Excel file with {output_size} bytes")
        
        if output_size == 0:
            return jsonify({"error": "Generated file is empty"}), 500
        
        # Clean up results file after successful creation
        try:
            os.unlink(results_file)
            session.pop('results_file', None)
            session.pop('total_processed', None)
        except Exception as e:
            print(f"[WARNING] Failed to delete results file: {str(e)}")
        
        # Return file with optimized headers
        response = send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
        # Add headers for better download performance
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response

    except Exception as e:
        print(f"[ERROR] Download failed: {str(e)}")
        return jsonify({"error": "Failed to generate download file"}), 500

@app.route("/download-results-stream", methods=["POST"])
def download_results_stream():
    """Streaming download for very large files"""
    try:
        # Get results file path from session
        results_file = session.get('results_file')
        if not results_file or not os.path.exists(results_file):
            return jsonify({"error": "No results available for download"}), 400

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'validation_results_{timestamp}.xlsx'
        
        def generate():
            """Generator function for streaming Excel data"""
            try:
                # Read JSON data
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                if not results_data:
                    yield b''
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(results_data)
                # Ensure all required columns are present and in order
                required_columns = [
                    'input_area', 'input_city', 'input_state', 'input_country', 'input_latitude', 'input_longitude', 'input_pincode',
                    'matched_area', 'matched_city', 'matched_state', 'matched_country', 'matched_latitude', 'matched_longitude', 'matched_pincode', 'valid', 'difference'
                ]
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = ''
                # Reorder columns
                df = df[required_columns]
                # Capitalize all string columns and rename for Excel
                df.columns = [col.upper() for col in df.columns]
                for col in df.select_dtypes(include='object').columns:
                    df[col] = df[col].astype(str).str.upper()
                
                # Create Excel in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Capitalize column names for the Excel output
                    df.columns = [col.upper() for col in df.columns]
                    df.to_excel(writer, index=False, sheet_name='Validation Results')
                    workbook = writer.book
                    worksheet = writer.sheets['Validation Results']
                    header_format = workbook.add_format({
                        'bold': True,
                        'border': 1
                    })
                    cell_format = workbook.add_format({'border': 1})
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column(col_num, col_num, 15)
                    if len(df) > 0:
                        worksheet.conditional_format(1, 0, len(df), len(df.columns) - 1, {'type': 'no_blanks', 'format': cell_format})
                        worksheet.conditional_format(1, 0, len(df), len(df.columns) - 1, {'type': 'blanks', 'format': cell_format})
                    if 'VALID' in df.columns:
                        valid_col_idx = df.columns.get_loc('VALID')
                        nrows = len(df) + 1  # +1 for header
                        worksheet.conditional_format(1, valid_col_idx, nrows, valid_col_idx, {
                            'type': 'cell',
                            'criteria': '==',
                            'value': True,
                            'format': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                        })
                        worksheet.conditional_format(1, valid_col_idx, nrows, valid_col_idx, {
                            'type': 'cell',
                            'criteria': '==',
                            'value': False,
                            'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                        })
                
                output.seek(0)
                yield output.getvalue()
                
                # Clean up
                try:
                    os.unlink(results_file)
                    session.pop('results_file', None)
                    session.pop('total_processed', None)
                except:
                    pass
                    
            except Exception as e:
                print(f"[ERROR] Streaming failed: {str(e)}")
                yield b''
        
        # Return streaming response
        response = app.response_class(
            generate(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
        return response

    except Exception as e:
        print(f"[ERROR] Streaming download failed: {str(e)}")
        return jsonify({"error": "Failed to generate download file"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
