import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

def train_bias_t_model(excel_file_path):
    """Train Bias-T model and save artifacts"""
    # Load data
    df = pd.read_excel(excel_file_path)

    # --- 1. Clean and preprocess ---

    # Frequency range split (GHz to MHz)
    def freq_split(freq_range):
        if pd.isna(freq_range): 
            return np.nan, np.nan
        freq_range = freq_range.replace("–", "-").replace("—", "-").strip()
        try:
            low, high = freq_range.split("-")
            return float(low.strip()) * 1000, float(high.strip()) * 1000  # GHz to MHz
        except:
            return np.nan, np.nan

    df[['Freq Low (MHz)', 'Freq High (MHz)']] = df['Freq Range (GHz)'].apply(lambda x: pd.Series(freq_split(x)))

    # Safe parser for dB fields
    def parse_db(value):
        if pd.isna(value):
            return np.nan
        try:
            val = str(value).replace("<", "").replace(">", "").replace("N/A", "").strip()
            return float(val) if val else np.nan
        except:
            return np.nan

    # Clean numeric fields
    df['Insertion Loss (dB)'] = df['Insertion Loss (dB)'].apply(parse_db)
    df['Return Loss (dB)'] = df['Return Loss (dB)'].apply(parse_db)
    df['Max DC Voltage (V)'] = df['Max DC Voltage (V)'].replace('N/A', np.nan).astype(float)
    df['Max DC Current (mA)'] = df['Max DC Current (mA)'].replace('N/A', np.nan).astype(float)

    # Add 'Unknown' row for label encoders (if not already present)
    for col in ['Connector Type', 'Manufacturer']:
        if 'Unknown' not in df[col].astype(str).unique():
            new_row = {c: np.nan for c in df.columns}
            new_row[col] = 'Unknown'
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save Part Number and Datasheet Link
    part_numbers = df['Part Number'].fillna("Unknown").values
    manufacturers = df['Manufacturer'].fillna("Unknown").values
    datasheet_urls = df['Datasheet Link'].fillna("Unknown").values
    freq_ranges = df['Freq Range (GHz)'].fillna("Unknown").values
    insertion_losses = df['Insertion Loss (dB)'].fillna(0).values
    return_losses = df['Return Loss (dB)'].fillna(0).values
    max_dc_voltages = df['Max DC Voltage (V)'].fillna(0).values
    max_dc_currents = df['Max DC Current (mA)'].fillna(0).values
    connector_types = df['Connector Type'].fillna("Unknown").values

    # --- 2. Feature selection ---
    drop_cols = ['Part Number', 'Datasheet Link', 'Freq Range (GHz)']
    X = df.drop(columns=drop_cols)

    # Columns
    num_cols = ['Freq Low (MHz)', 'Freq High (MHz)', 'Insertion Loss (dB)',
                'Return Loss (dB)', 'Max DC Voltage (V)', 'Max DC Current (mA)']
    cat_cols = ['Connector Type', 'Manufacturer']

    # Fill numeric with median
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Fill categorical with 'Unknown'
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown').astype(str)

    # Encode categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Scale numeric columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Final NaN check
    if X.isnull().values.any():
        print("⚠️ Warning: NaNs found, applying final fill.")
        X = X.fillna(0)

    # Debug info
    print("✅ All NaNs removed. Training data shape:", X.shape)

    # --- 3. Model training ---
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)

    # --- 4. Save artifacts ---
    joblib.dump(nn, 'models/bias_t_nn_model.joblib')
    joblib.dump(scaler, 'models/bias_t_scaler.joblib')
    for col in cat_cols:
        joblib.dump(label_encoders[col], f'models/bias_t_le_{col}.joblib')
    joblib.dump(part_numbers, 'models/bias_t_part_numbers.joblib')
    joblib.dump(manufacturers, 'models/bias_t_manufacturers.joblib')
    joblib.dump(datasheet_urls, 'models/bias_t_datasheet_urls.joblib')
    joblib.dump(freq_ranges, 'models/bias_t_freq_ranges.joblib')
    joblib.dump(insertion_losses, 'models/bias_t_insertion_losses.joblib')
    joblib.dump(return_losses, 'models/bias_t_return_losses.joblib')
    joblib.dump(max_dc_voltages, 'models/bias_t_max_dc_voltages.joblib')
    joblib.dump(max_dc_currents, 'models/bias_t_max_dc_currents.joblib')
    joblib.dump(connector_types, 'models/bias_t_connector_types.joblib')
    joblib.dump(X.columns.tolist(), 'models/bias_t_feature_cols.joblib')

    print("✅ Bias-T NearestNeighbors model and encoders saved successfully.")

def recommend_bias_t(input_features, top_k=1):
    """Recommend Bias-T components based on input features"""
    try:
        # Load models and encoders
        bias_nn = joblib.load('models/bias_t_nn_model.joblib')
        bias_scaler = joblib.load('models/bias_t_scaler.joblib')
        bias_part_numbers = joblib.load('models/bias_t_part_numbers.joblib')
        bias_manufacturers = joblib.load('models/bias_t_manufacturers.joblib')
        bias_datasheet_urls = joblib.load('models/bias_t_datasheet_urls.joblib')
        bias_freq_ranges = joblib.load('models/bias_t_freq_ranges.joblib')
        bias_insertion_losses = joblib.load('models/bias_t_insertion_losses.joblib')
        bias_return_losses = joblib.load('models/bias_t_return_losses.joblib')
        bias_max_dc_voltages = joblib.load('models/bias_t_max_dc_voltages.joblib')
        bias_max_dc_currents = joblib.load('models/bias_t_max_dc_currents.joblib')
        bias_connector_types = joblib.load('models/bias_t_connector_types.joblib')
        bias_feature_cols = joblib.load('models/bias_t_feature_cols.joblib')
        le_connector = joblib.load('models/bias_t_le_Connector Type.joblib')
        le_manuf_bias = joblib.load('models/bias_t_le_Manufacturer.joblib')
    except Exception as e:
        print(f"Error loading models: {e}")
        return []

    num_cols_bias = ['Freq Low (MHz)', 'Freq High (MHz)', 'Insertion Loss (dB)',
                     'Return Loss (dB)', 'Max DC Voltage (V)', 'Max DC Current (mA)']
    cat_cols_bias = ['Connector Type', 'Manufacturer']

    # Create input DataFrame with default values
    input_df = pd.DataFrame([{
        'Freq Low (MHz)': input_features.get('freq_low', 0) * 1000,  # Convert GHz to MHz
        'Freq High (MHz)': input_features.get('freq_high', 0) * 1000,  # Convert GHz to MHz
        'Insertion Loss (dB)': 0,
        'Return Loss (dB)': 0,
        'Max DC Voltage (V)': 0,
        'Max DC Current (mA)': 0,
        'Connector Type': 'Unknown',
        'Manufacturer': 'Unknown'
    }])
    
    # Ensure all required columns exist
    for col in bias_feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Handle categorical columns
    for col in cat_cols_bias:
        if col in input_df.columns:
            val = input_df.at[0, col]
            # Handle NaN or missing values
            if pd.isna(val) or str(val).strip() == '':
                val = 'Unknown'
            # Handle values not in encoder classes
            try:
                if col == 'Connector Type':
                    if val not in le_connector.classes_:
                        val = 'Unknown'
                else:  # Manufacturer
                    if val not in le_manuf_bias.classes_:
                        val = 'Unknown'
            except:
                val = 'Unknown'
            input_df.at[0, col] = val
        else:
            input_df[col] = 'Unknown'

    # Encode categorical columns
    input_df['Connector Type'] = le_connector.transform(input_df['Connector Type'])
    input_df['Manufacturer'] = le_manuf_bias.transform(input_df['Manufacturer'])

    # Handle numeric columns - ensure they are numeric and fill NaNs
    for col in num_cols_bias:
        if col in input_df.columns:
            try:
                # Convert to numeric, coercing errors to NaN
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                # Fill NaN with 0
                input_df[col] = input_df[col].fillna(0)
            except:
                input_df[col] = 0
        else:
            input_df[col] = 0

    # Reorder columns exactly as training
    input_df = input_df[bias_feature_cols]

    # Final safety check - fill any remaining NaNs
    input_df = input_df.fillna(0)

    # Scale numeric columns with error handling
    try:
        input_df[num_cols_bias] = bias_scaler.transform(input_df[num_cols_bias])
    except Exception as e:
        print(f"⚠️ Scaling failed, using original values: {e}")
        # If scaling fails, keep original values but ensure they're finite
        for col in num_cols_bias:
            input_df[col] = input_df[col].replace([np.inf, -np.inf], 0)
            input_df[col] = input_df[col].fillna(0)

    # Final NaN check and cleanup
    nan_cols = input_df.columns[input_df.isna().any()].tolist()
    if nan_cols:
        print(f"⚠️ NaNs found after scaling in columns: {nan_cols}")
        # Replace NaNs with 0
        input_df = input_df.fillna(0)
        # Also replace any infinite values
        input_df = input_df.replace([np.inf, -np.inf], 0)

    distances, indices = bias_nn.kneighbors(input_df, n_neighbors=top_k)

    recommendations = []
    for idx in indices[0]:
        try:
            # Clean and encode strings to handle special characters
            def clean_string(value):
                if isinstance(value, str):
                    # Replace problematic characters
                    value = value.replace('–', '-').replace('—', '-')
                    # Encode to handle any remaining special characters
                    try:
                        return value.encode('utf-8', errors='ignore').decode('utf-8')
                    except:
                        return str(value)
                return str(value)
            
            recommendations.append({
                'Part Number': clean_string(bias_part_numbers[idx]),
                'Manufacturer': clean_string(bias_manufacturers[idx]),
                'Frequency Range': clean_string(bias_freq_ranges[idx]),
                'Insertion Loss': f"{bias_insertion_losses[idx]} dB" if bias_insertion_losses[idx] != 0 else "N/A",
                'Return Loss': f"{bias_return_losses[idx]} dB" if bias_return_losses[idx] != 0 else "N/A",
                'Max DC Voltage': f"{bias_max_dc_voltages[idx]} V" if bias_max_dc_voltages[idx] != 0 else "N/A",
                'Max DC Current': f"{bias_max_dc_currents[idx]} mA" if bias_max_dc_currents[idx] != 0 else "N/A",
                'Connector Type': clean_string(bias_connector_types[idx]),
                'Datasheet URL': clean_string(bias_datasheet_urls[idx])
            })
        except Exception as e:
            print(f"Error creating recommendation {idx}: {e}")
            # Add a fallback recommendation
            recommendations.append({
                'Part Number': 'Unknown',
                'Manufacturer': 'Unknown',
                'Frequency Range': 'N/A',
                'Insertion Loss': 'N/A',
                'Return Loss': 'N/A',
                'Max DC Voltage': 'N/A',
                'Max DC Current': 'N/A',
                'Connector Type': 'Unknown',
                'Datasheet URL': 'N/A'
            })

    return recommendations 