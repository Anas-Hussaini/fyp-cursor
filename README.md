# RF Component Recommendation API

A FastAPI service for recommending RF components (LNA and Bias-T) based on technical specifications.

## Features

- **LNA Recommendations**: Get Low Noise Amplifier recommendations based on frequency, gain, noise figure, and other specifications
- **Bias-T Recommendations**: Get Bias-T component recommendations based on frequency range, voltage, current, and loss parameters
- **RESTful API**: Clean REST endpoints with automatic documentation
- **Input Validation**: Robust input validation using Pydantic models
- **Error Handling**: Comprehensive error handling and meaningful error messages

## Project Structure

```
fyp-cursor/
├── main.py              # FastAPI application
├── models.py            # Pydantic models for request/response validation
├── lna_model.py         # LNA model training and prediction logic
├── bias_t_model.py      # Bias-T model training and prediction logic
├── train_models.py      # Script to train both models
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── models/             # Directory for saved model files (created automatically)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data Files

Place your Excel files in the project root:
- `LNAs.xlsx` - LNA component data
- `BiasTs.xlsx` - Bias-T component data

### 3. Train Models

```bash
python train_models.py
```

This will:
- Create a `models/` directory
- Train the LNA model using Random Forest classifier
- Train the Bias-T model using Nearest Neighbors
- Save all model artifacts and encoders

### 4. Run the API Server

```bash
python main.py
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. LNA Recommendation

**Endpoint**: `POST /recommend-lna`

**Request Body**:
```json
{
  "freq_low": 1.0,
  "freq_high": 2.0,
  "noise_figure_db": 2.5,
  "gain_db": 20.0,
  "supply_voltage_v": 5.0,
  "supply_current_ma": 50.0,
  "input_return_loss_db": -15.0,
  "output_return_loss_db": -15.0,
  "s11_db": -15.0,
  "s21_db": 20.0,
  "s12_db": -30.0,
  "s22_db": -15.0,
  "package_type": "SOT-89",
  "key_notes": "Low noise"
}
```

**Response**:
```json
{
  "part_number": "LNA123",
  "manufacturer": "Example Corp",
  "datasheet_url": "https://example.com/datasheet.pdf"
}
```

### 2. Bias-T Recommendation

**Endpoint**: `POST /recommend-bias-t`

**Request Body**:
```json
{
  "freq_low_mhz": 1000.0,
  "freq_high_mhz": 2000.0,
  "max_dc_voltage_v": 12.0,
  "max_current_ma": 100.0,
  "insertion_loss_db": 0.5,
  "return_loss_db": -20.0,
  "connector_type": "SMA",
  "manufacturer": "Example Corp"
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "Part Number": "BT456",
      "Datasheet URL": "https://example.com/bias-t-datasheet.pdf"
    }
  ]
}
```

## Usage Examples

### Using curl

**LNA Recommendation**:
```bash
curl -X POST "http://localhost:8000/recommend-lna" \
     -H "Content-Type: application/json" \
     -d '{
       "freq_low": 1.0,
       "freq_high": 2.0,
       "noise_figure_db": 2.5,
       "gain_db": 20.0
     }'
```

**Bias-T Recommendation**:
```bash
curl -X POST "http://localhost:8000/recommend-bias-t" \
     -H "Content-Type: application/json" \
     -d '{
       "freq_low_mhz": 1000.0,
       "freq_high_mhz": 2000.0,
       "max_dc_voltage_v": 12.0,
       "max_current_ma": 100.0,
       "insertion_loss_db": 0.5,
       "return_loss_db": -20.0,
       "connector_type": "SMA",
       "manufacturer": "Example Corp"
     }'
```

### Using Python requests

```python
import requests

# LNA recommendation
lna_data = {
    "freq_low": 1.0,
    "freq_high": 2.0,
    "noise_figure_db": 2.5,
    "gain_db": 20.0
}

response = requests.post("http://localhost:8000/recommend-lna", json=lna_data)
print(response.json())

# Bias-T recommendation
bias_t_data = {
    "freq_low_mhz": 1000.0,
    "freq_high_mhz": 2000.0,
    "max_dc_voltage_v": 12.0,
    "max_current_ma": 100.0,
    "insertion_loss_db": 0.5,
    "return_loss_db": -20.0,
    "connector_type": "SMA",
    "manufacturer": "Example Corp"
}

response = requests.post("http://localhost:8000/recommend-bias-t", json=bias_t_data)
print(response.json())
```

## Data Format Requirements

### LNA Excel File (`LNAs.xlsx`)
Required columns:
- `Part Number`
- `Manufacturer`
- `Datasheet Link`
- `Freq Range (GHz)` (format: "1.0-2.0")
- `Gain (dB)`
- `Noise Figure (dB)`
- `Connector Type` (optional)

### Bias-T Excel File (`BiasTs.xlsx`)
Required columns:
- `Part Number`
- `Manufacturer`
- `Datasheet Link`
- `Freq Range (GHz)` (format: "1.0-2.0")
- `Insertion Loss (dB)`
- `Return Loss (dB)`
- `Max DC Voltage (V)`
- `Max DC Current (mA)`
- `Connector Type`

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Model loading or prediction errors
- **404 Not Found**: Invalid endpoints

All errors return JSON responses with descriptive messages.

## Development

### Adding New Features

1. **New Endpoints**: Add to `main.py`
2. **New Models**: Create new model files following the pattern of `lna_model.py`
3. **New Data Types**: Update `models.py` with new Pydantic models

### Testing

The API includes automatic documentation at `/docs` where you can test endpoints interactively.

### Production Deployment

For production deployment:

1. Set `reload=False` in `main.py`
2. Configure proper CORS origins
3. Use a production ASGI server like Gunicorn
4. Set up proper logging and monitoring

## Troubleshooting

### Common Issues

1. **Model files not found**: Run `python train_models.py` first
2. **Excel file errors**: Ensure your Excel files have the required columns
3. **Port already in use**: Change the port in `main.py` or kill the existing process

### Logs

Check the console output for detailed error messages and training progress.

## License

This project is for educational/research purposes. 