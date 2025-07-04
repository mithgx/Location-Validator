﻿# Location Validator

Location Validator is an offline, privacy-focused application for validating and processing large-scale location data without relying on any third-party services. It enables users to upload Excel files containing area names, pincodes, and latitude/longitude information, and rapidly verifies the accuracy of these entries against a comprehensive, locally stored global postal dataset.

## Key Features

- **Offline Processing:** All validation is performed locally, ensuring data privacy and security—no internet or external API calls required.
- **Batch Validation:** Upload Excel files with thousands of records and receive instant feedback on the validity of each entry.
- **Flexible Search:** Instantly search for location details by area name or pincode.
- **Detailed Results:** Download color-coded Excel reports highlighting valid and invalid entries for easy review.
- **Interactive Web Interface:** Simple, user-friendly interface for uploading files, mapping columns, and viewing results.

## Methodology

- The app normalizes and cleans input data, then cross-references it with a cached, pre-processed global postal code database.
- Validation checks include matching area names and pincodes, as well as verifying latitude/longitude proximity.
- All logic is optimized for speed and memory efficiency, enabling the processing of large datasets on standard hardware.

## Getting Started

### Prerequisites
- Python 3.8+
- See `requirements.txt` for required Python packages

### Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `data/allCountries.parquet` file is present in the `data/` directory.

### Running the App
```bash
python app.py
```

### Usage
1. **Upload Data:** Use the web interface to upload your Excel file containing location data.
2. **Map Columns:** Map your file's columns to the required fields (area, pincode, latlong).
3. **Validate:** Start the validation process and view a summary of results directly in the browser.
4. **Download Results:** Download a detailed Excel report with validation status for each entry.

---

