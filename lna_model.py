import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

def train_lna_model(excel_file_path):
    """Train LNA model and save artifacts"""
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
    
    # Clean numeric fields - ensure they are numeric
    df['Gain (dB)'] = pd.to_numeric(df['Gain (dB)'], errors='coerce')
    df['Noise Figure (dB)'] = pd.to_numeric(df['Noise Figure (dB)'], errors='coerce')
    
    # Fill NaN values in numeric columns
    df['Gain (dB)'] = df['Gain (dB)'].fillna(0)
    df['Noise Figure (dB)'] = df['Noise Figure (dB)'].fillna(0)
    df['Freq Low (MHz)'] = df['Freq Low (MHz)'].fillna(0)
    df['Freq High (MHz)'] = df['Freq High (MHz)'].fillna(0)
    
    # Save Part Number, Manufacturer, and Datasheet Link
    part_numbers = df['Part Number'].fillna("Unknown").values
    manufacturers = df['Manufacturer'].fillna("Unknown").values
    datasheet_urls = df['Datasheet Link'].fillna("Unknown").values
    freq_ranges = df['Freq Range (GHz)'].fillna("Unknown").values
    gains = df['Gain (dB)'].values
    noise_figures = df['Noise Figure (dB)'].values
    connector_types = df['Connector Type'].fillna("Unknown").values
    
    # --- 2. Feature selection ---
    drop_cols = ['Part Number', 'Manufacturer', 'Datasheet Link', 'Freq Range (GHz)', 'Connector Type']
    X = df.drop(columns=drop_cols)
    
    # Ensure all remaining columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # --- 3. Model training ---
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    
    # --- 4. Save artifacts ---
    joblib.dump(nn, 'models/lna_nn_model.joblib')
    joblib.dump(part_numbers, 'models/lna_part_numbers.joblib')
    joblib.dump(manufacturers, 'models/lna_manufacturers.joblib')
    joblib.dump(datasheet_urls, 'models/lna_datasheet_urls.joblib')
    joblib.dump(freq_ranges, 'models/lna_freq_ranges.joblib')
    joblib.dump(gains, 'models/lna_gains.joblib')
    joblib.dump(noise_figures, 'models/lna_noise_figures.joblib')
    joblib.dump(connector_types, 'models/lna_connector_types.joblib')
    joblib.dump(X.columns.tolist(), 'models/lna_feature_cols.joblib')
    
    print("✅ LNA model trained and saved successfully!")

def predict_lna(input_features):
    """Predict LNA components based on input features"""
    # Load model and data
    lna_nn = joblib.load('models/lna_nn_model.joblib')
    lna_part_numbers = joblib.load('models/lna_part_numbers.joblib')
    lna_manufacturers = joblib.load('models/lna_manufacturers.joblib')
    lna_datasheet_urls = joblib.load('models/lna_datasheet_urls.joblib')
    lna_freq_ranges = joblib.load('models/lna_freq_ranges.joblib')
    lna_gains = joblib.load('models/lna_gains.joblib')
    lna_noise_figures = joblib.load('models/lna_noise_figures.joblib')
    lna_connector_types = joblib.load('models/lna_connector_types.joblib')
    lna_feature_cols = joblib.load('models/lna_feature_cols.joblib')
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_features])
    
    # Ensure all required columns exist
    for col in lna_feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns exactly as training
    input_df = input_df[lna_feature_cols]
    
    # Make prediction
    distances, indices = lna_nn.kneighbors(input_df, n_neighbors=1)
    
    # Get the best match
    idx = indices[0][0]
    
    # Format the response
    return {
        'part_number': lna_part_numbers[idx],
        'manufacturer': lna_manufacturers[idx],
        'frequency_range': lna_freq_ranges[idx],
        'gain': f"{lna_gains[idx]} dB" if lna_gains[idx] != 0 else "N/A",
        'noise_figure': f"{lna_noise_figures[idx]} dB" if lna_noise_figures[idx] != 0 else "N/A",
        'package': lna_connector_types[idx],
        'datasheet_url': lna_datasheet_urls[idx]
    } 