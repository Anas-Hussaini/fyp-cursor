# RF Component Recommendation Service - Complete Pseudocode

## 1. SYSTEM OVERVIEW

```
RF Component Recommendation Service
├── FastAPI Web Server (main.py)
├── Machine Learning Models
│   ├── LNA Recommendation Model (lna_model.py)
│   └── Bias-T Recommendation Model (bias_t_model.py)
├── Natural Language Processing (nlp_extractor.py)
├── Conversational AI Manager (chat_manager.py)
├── Data Models (models.py)
└── Web Interface (static/index.html)
```

## 2. MAIN APPLICATION (main.py)

### 2.1 Application Initialization
```
FUNCTION initialize_application():
    // Load environment variables
    load_dotenv()
    
    // Create FastAPI app with metadata
    app = FastAPI(
        title: "RF Component Recommendation API",
        description: "API for recommending LNA and Bias-T components",
        version: "1.0.0"
    )
    
    // Add CORS middleware for cross-origin requests
    app.add_middleware(CORS, allow_origins: ["*"])
    
    // Mount static files for web interface
    app.mount("/static", StaticFiles)
    
    RETURN app
```

### 2.2 API Endpoints

#### Root Endpoint
```
ENDPOINT GET "/":
    RETURN {
        message: "RF Component Recommendation API",
        version: "1.0.0",
        endpoints: {
            health: "/health",
            chat: "/chat",
            lna_recommendation: "/recommend-lna",
            bias_t_recommendation: "/recommend-bias-t",
            extract_requirements: "/extract-requirements",
            web_interface: "/interface",
            docs: "/docs"
        }
    }
```

#### Health Check
```
ENDPOINT GET "/health":
    RETURN {status: "healthy", message: "API is running"}
```

#### Chat Endpoint
```
ENDPOINT POST "/chat":
    INPUT: ChatRequest {message: string, conversation_id: optional}
    
    TRY:
        response = conversation_manager.process_message(request)
        RETURN response
    CATCH Exception:
        RAISE HTTPException(500, "Chat error: " + str(e))
```

#### LNA Recommendation
```
ENDPOINT POST "/recommend-lna":
    INPUT: LNAInput {
        freq_low: float,
        freq_high: float,
        noise_figure_db: optional float,
        gain_db: optional float
    }
    
    TRY:
        input_features = input_data.model_dump(exclude_none: true)
        result = predict_lna(input_features)
        
        lna_rec = LNARecommendation(
            part_number: result.part_number,
            manufacturer: result.manufacturer,
            frequency_range: result.frequency_range,
            gain: result.gain,
            noise_figure: result.noise_figure,
            package: result.package,
            datasheet_url: result.datasheet_url
        )
        
        RETURN RecommendationResponse(recommendations: [lna_rec])
        
    CATCH Exception:
        RAISE HTTPException(500, "Error during LNA recommendation")
```

## 3. DATA MODELS (models.py)

### 3.1 Input Models
```
CLASS LNAInput:
    freq_low: float
    freq_high: float
    noise_figure_db: optional float
    gain_db: optional float

CLASS BiasTInput:
    freq_low: float
    freq_high: float

CLASS ChatRequest:
    message: string
    conversation_id: optional string
```

### 3.2 Response Models
```
CLASS LNARecommendation:
    part_number: string
    manufacturer: string
    frequency_range: string
    gain: string
    noise_figure: string
    package: string
    datasheet_url: string

CLASS BiasTRecommendation:
    part_number: string
    manufacturer: string
    frequency_range: string
    insertion_loss: string
    return_loss: string
    max_dc_voltage: string
    max_dc_current: string
    connector_type: string
    datasheet_url: string

CLASS ChatResponse:
    response: string
    conversation_id: string
    recommendations: optional dict
```

## 4. LNA MODEL (lna_model.py)

### 4.1 Model Training
```
FUNCTION train_lna_model(excel_file_path):
    // Load data from Excel file
    df = read_excel(excel_file_path)
    
    // 1. Clean and preprocess data
    FOR each row IN df:
        // Split frequency range (GHz to MHz)
        freq_range = row['Freq Range (GHz)']
        IF freq_range is not null:
            low, high = split_frequency_range(freq_range)
            row['Freq Low (MHz)'] = low * 1000
            row['Freq High (MHz)'] = high * 1000
        
        // Clean numeric fields
        row['Gain (dB)'] = to_numeric(row['Gain (dB)'])
        row['Noise Figure (dB)'] = to_numeric(row['Noise Figure (dB)'])
    
    // Fill NaN values
    df['Gain (dB)'] = fill_na(df['Gain (dB)'], 0)
    df['Noise Figure (dB)'] = fill_na(df['Noise Figure (dB)'], 0)
    df['Freq Low (MHz)'] = fill_na(df['Freq Low (MHz)'], 0)
    df['Freq High (MHz)'] = fill_na(df['Freq High (MHz)'], 0)
    
    // Save metadata
    part_numbers = df['Part Number'].fill_na("Unknown")
    manufacturers = df['Manufacturer'].fill_na("Unknown")
    datasheet_urls = df['Datasheet Link'].fill_na("Unknown")
    freq_ranges = df['Freq Range (GHz)'].fill_na("Unknown")
    gains = df['Gain (dB)']
    noise_figures = df['Noise Figure (dB)']
    connector_types = df['Connector Type'].fill_na("Unknown")
    
    // 2. Feature selection
    drop_columns = ['Part Number', 'Manufacturer', 'Datasheet Link', 'Freq Range (GHz)', 'Connector Type']
    X = df.drop(columns: drop_columns)
    
    // Ensure all columns are numeric
    FOR column IN X.columns:
        X[column] = to_numeric(X[column]).fill_na(0)
    
    // 3. Model training
    nn = NearestNeighbors(n_neighbors: 1)
    nn.fit(X)
    
    // 4. Save artifacts
    save_model(nn, 'models/lna_nn_model.joblib')
    save_data(part_numbers, 'models/lna_part_numbers.joblib')
    save_data(manufacturers, 'models/lna_manufacturers.joblib')
    save_data(datasheet_urls, 'models/lna_datasheet_urls.joblib')
    save_data(freq_ranges, 'models/lna_freq_ranges.joblib')
    save_data(gains, 'models/lna_gains.joblib')
    save_data(noise_figures, 'models/lna_noise_figures.joblib')
    save_data(connector_types, 'models/lna_connector_types.joblib')
    save_data(X.columns, 'models/lna_feature_cols.joblib')
    
    PRINT "✅ LNA model trained and saved successfully!"
```

### 4.2 LNA Prediction
```
FUNCTION predict_lna(input_features):
    // Load model and data
    lna_nn = load_model('models/lna_nn_model.joblib')
    lna_part_numbers = load_data('models/lna_part_numbers.joblib')
    lna_manufacturers = load_data('models/lna_manufacturers.joblib')
    lna_datasheet_urls = load_data('models/lna_datasheet_urls.joblib')
    lna_freq_ranges = load_data('models/lna_freq_ranges.joblib')
    lna_gains = load_data('models/lna_gains.joblib')
    lna_noise_figures = load_data('models/lna_noise_figures.joblib')
    lna_connector_types = load_data('models/lna_connector_types.joblib')
    lna_feature_cols = load_data('models/lna_feature_cols.joblib')
    
    // Create input DataFrame
    input_df = create_dataframe([input_features])
    
    // Ensure all required columns exist
    FOR column IN lna_feature_cols:
        IF column not in input_df.columns:
            input_df[column] = 0
    
    // Reorder columns exactly as training
    input_df = input_df[lna_feature_cols]
    
    // Make prediction
    distances, indices = lna_nn.kneighbors(input_df, n_neighbors: 1)
    
    // Get the best match
    idx = indices[0][0]
    
    // Format the response
    RETURN {
        part_number: lna_part_numbers[idx],
        manufacturer: lna_manufacturers[idx],
        frequency_range: lna_freq_ranges[idx],
        gain: format_gain(lna_gains[idx]),
        noise_figure: format_noise_figure(lna_noise_figures[idx]),
        package: lna_connector_types[idx],
        datasheet_url: lna_datasheet_urls[idx]
    }
```

## 5. BIAS-T MODEL (bias_t_model.py)

### 5.1 Model Training
```
FUNCTION train_bias_t_model(excel_file_path):
    // Load data from Excel file
    df = read_excel(excel_file_path)
    
    // 1. Clean and preprocess data
    FOR each row IN df:
        // Split frequency range (GHz to MHz)
        freq_range = row['Freq Range (GHz)']
        IF freq_range is not null:
            low, high = split_frequency_range(freq_range)
            row['Freq Low (MHz)'] = low * 1000
            row['Freq High (MHz)'] = high * 1000
        
        // Parse dB fields safely
        row['Insertion Loss (dB)'] = parse_db_field(row['Insertion Loss (dB)'])
        row['Return Loss (dB)'] = parse_db_field(row['Return Loss (dB)'])
        row['Max DC Voltage (V)'] = to_numeric(row['Max DC Voltage (V)'])
        row['Max DC Current (mA)'] = to_numeric(row['Max DC Current (mA)'])
    
    // Add 'Unknown' row for label encoders
    FOR column IN ['Connector Type', 'Manufacturer']:
        IF 'Unknown' not in df[column].unique():
            new_row = create_empty_row(df.columns)
            new_row[column] = 'Unknown'
            df = concatenate(df, new_row)
    
    // Save metadata
    part_numbers = df['Part Number'].fill_na("Unknown")
    manufacturers = df['Manufacturer'].fill_na("Unknown")
    datasheet_urls = df['Datasheet Link'].fill_na("Unknown")
    freq_ranges = df['Freq Range (GHz)'].fill_na("Unknown")
    insertion_losses = df['Insertion Loss (dB)'].fill_na(0)
    return_losses = df['Return Loss (dB)'].fill_na(0)
    max_dc_voltages = df['Max DC Voltage (V)'].fill_na(0)
    max_dc_currents = df['Max DC Current (mA)'].fill_na(0)
    connector_types = df['Connector Type'].fill_na("Unknown")
    
    // 2. Feature selection
    drop_columns = ['Part Number', 'Datasheet Link', 'Freq Range (GHz)']
    X = df.drop(columns: drop_columns)
    
    // Define column types
    numeric_columns = ['Freq Low (MHz)', 'Freq High (MHz)', 'Insertion Loss (dB)',
                      'Return Loss (dB)', 'Max DC Voltage (V)', 'Max DC Current (mA)']
    categorical_columns = ['Connector Type', 'Manufacturer']
    
    // Fill numeric with median
    FOR column IN numeric_columns:
        X[column] = to_numeric(X[column])
        median_val = X[column].median()
        X[column] = X[column].fill_na(median_val)
    
    // Fill categorical with 'Unknown'
    FOR column IN categorical_columns:
        X[column] = X[column].fill_na('Unknown')
    
    // Encode categorical columns
    label_encoders = {}
    FOR column IN categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    // Scale numeric columns
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    // Final NaN check
    IF X.has_nulls():
        X = X.fill_na(0)
    
    // 3. Model training
    nn = NearestNeighbors(n_neighbors: 5)
    nn.fit(X)
    
    // 4. Save artifacts
    save_model(nn, 'models/bias_t_nn_model.joblib')
    save_model(scaler, 'models/bias_t_scaler.joblib')
    FOR column IN categorical_columns:
        save_model(label_encoders[column], f'models/bias_t_le_{column}.joblib')
    save_data(part_numbers, 'models/bias_t_part_numbers.joblib')
    save_data(manufacturers, 'models/bias_t_manufacturers.joblib')
    save_data(datasheet_urls, 'models/bias_t_datasheet_urls.joblib')
    save_data(freq_ranges, 'models/bias_t_freq_ranges.joblib')
    save_data(insertion_losses, 'models/bias_t_insertion_losses.joblib')
    save_data(return_losses, 'models/bias_t_return_losses.joblib')
    save_data(max_dc_voltages, 'models/bias_t_max_dc_voltages.joblib')
    save_data(max_dc_currents, 'models/bias_t_max_dc_currents.joblib')
    save_data(connector_types, 'models/bias_t_connector_types.joblib')
    save_data(X.columns, 'models/bias_t_feature_cols.joblib')
    
    PRINT "✅ Bias-T NearestNeighbors model and encoders saved successfully."
```

### 5.2 Bias-T Recommendation
```
FUNCTION recommend_bias_t(input_features, top_k: 1):
    TRY:
        // Load models and encoders
        bias_nn = load_model('models/bias_t_nn_model.joblib')
        bias_scaler = load_model('models/bias_t_scaler.joblib')
        bias_part_numbers = load_data('models/bias_t_part_numbers.joblib')
        bias_manufacturers = load_data('models/bias_t_manufacturers.joblib')
        bias_datasheet_urls = load_data('models/bias_t_datasheet_urls.joblib')
        bias_freq_ranges = load_data('models/bias_t_freq_ranges.joblib')
        bias_insertion_losses = load_data('models/bias_t_insertion_losses.joblib')
        bias_return_losses = load_data('models/bias_t_return_losses.joblib')
        bias_max_dc_voltages = load_data('models/bias_t_max_dc_voltages.joblib')
        bias_max_dc_currents = load_data('models/bias_t_max_dc_currents.joblib')
        bias_connector_types = load_data('models/bias_t_connector_types.joblib')
        bias_feature_cols = load_data('models/bias_t_feature_cols.joblib')
        le_connector = load_model('models/bias_t_le_Connector Type.joblib')
        le_manuf_bias = load_model('models/bias_t_le_Manufacturer.joblib')
    CATCH Exception:
        PRINT "Error loading models: " + str(e)
        RETURN []
    
    // Define column types
    numeric_columns = ['Freq Low (MHz)', 'Freq High (MHz)', 'Insertion Loss (dB)',
                      'Return Loss (dB)', 'Max DC Voltage (V)', 'Max DC Current (mA)']
    categorical_columns = ['Connector Type', 'Manufacturer']
    
    // Create input DataFrame with default values
    input_df = create_dataframe([{
        'Freq Low (MHz)': input_features.get('freq_low', 0) * 1000,
        'Freq High (MHz)': input_features.get('freq_high', 0) * 1000,
        'Insertion Loss (dB)': 0,
        'Return Loss (dB)': 0,
        'Max DC Voltage (V)': 0,
        'Max DC Current (mA)': 0,
        'Connector Type': 'Unknown',
        'Manufacturer': 'Unknown'
    }])
    
    // Ensure all required columns exist
    FOR column IN bias_feature_cols:
        IF column not in input_df.columns:
            input_df[column] = 0
    
    // Handle categorical columns
    FOR column IN categorical_columns:
        IF column in input_df.columns:
            val = input_df.at[0, column]
            IF val is null OR val.strip() == '':
                val = 'Unknown'
            TRY:
                IF column == 'Connector Type':
                    IF val not in le_connector.classes_:
                        val = 'Unknown'
                ELSE: // Manufacturer
                    IF val not in le_manuf_bias.classes_:
                        val = 'Unknown'
            CATCH:
                val = 'Unknown'
            input_df.at[0, column] = val
        ELSE:
            input_df[column] = 'Unknown'
    
    // Encode categorical columns
    input_df['Connector Type'] = le_connector.transform(input_df['Connector Type'])
    input_df['Manufacturer'] = le_manuf_bias.transform(input_df['Manufacturer'])
    
    // Handle numeric columns
    FOR column IN numeric_columns:
        IF column in input_df.columns:
            TRY:
                input_df[column] = to_numeric(input_df[column])
                input_df[column] = input_df[column].fill_na(0)
            CATCH:
                input_df[column] = 0
        ELSE:
            input_df[column] = 0
    
    // Reorder columns exactly as training
    input_df = input_df[bias_feature_cols]
    
    // Final safety check
    input_df = input_df.fill_na(0)
    
    // Scale numeric columns
    TRY:
        input_df[numeric_columns] = bias_scaler.transform(input_df[numeric_columns])
    CATCH Exception:
        PRINT "⚠️ Scaling failed, using original values"
        FOR column IN numeric_columns:
            input_df[column] = input_df[column].replace([inf, -inf], 0)
            input_df[column] = input_df[column].fill_na(0)
    
    // Final NaN check and cleanup
    IF input_df.has_nulls():
        input_df = input_df.fill_na(0)
        input_df = input_df.replace([inf, -inf], 0)
    
    // Make prediction
    distances, indices = bias_nn.kneighbors(input_df, n_neighbors: top_k)
    
    // Format recommendations
    recommendations = []
    FOR idx IN indices[0]:
        TRY:
            recommendations.append({
                'Part Number': clean_string(bias_part_numbers[idx]),
                'Manufacturer': clean_string(bias_manufacturers[idx]),
                'Frequency Range': clean_string(bias_freq_ranges[idx]),
                'Insertion Loss': format_insertion_loss(bias_insertion_losses[idx]),
                'Return Loss': format_return_loss(bias_return_losses[idx]),
                'Max DC Voltage': format_dc_voltage(bias_max_dc_voltages[idx]),
                'Max DC Current': format_dc_current(bias_max_dc_currents[idx]),
                'Connector Type': clean_string(bias_connector_types[idx]),
                'Datasheet URL': clean_string(bias_datasheet_urls[idx])
            })
        CATCH Exception:
            recommendations.append(create_fallback_recommendation())
    
    RETURN recommendations
```

## 6. NLP EXTRACTOR (nlp_extractor.py)

### 6.1 Regex Fallback Extraction
```
FUNCTION _regex_extract(user_request: string):
    extracted = {
        freq_low: null,
        freq_high: null,
        gain_db: null,
        noise_figure_db: null
    }
    
    // Extract frequency range
    range_match = search_regex(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(GHz|MHz)", user_request)
    IF range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        unit = range_match.group(3).lower()
        IF unit == "mhz":
            low /= 1000.0
            high /= 1000.0
        extracted.freq_low = low
        extracted.freq_high = high
    ELSE:
        single_match = search_regex(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)", user_request)
        IF single_match:
            value = float(single_match.group(1))
            unit = single_match.group(2).lower()
            IF unit == "mhz":
                value /= 1000.0
            extracted.freq_low = value
            extracted.freq_high = value
    
    // Extract gain
    gain_match = search_regex(r"(\d+\.?\d*)\s*dB.*gain", user_request)
    IF gain_match:
        extracted.gain_db = float(gain_match.group(1))
    
    // Extract noise figure
    noise_match = search_regex(r"(\d+\.?\d*)\s*dB.*noise", user_request)
    IF noise_match:
        extracted.noise_figure_db = float(noise_match.group(1))
    
    RETURN extracted
```

### 6.2 OpenAI Extraction
```
FUNCTION extract_requirements_via_openai(user_request: string):
    // Check for API key
    api_key = get_env("OPENAI_API_KEY")
    IF not api_key:
        RETURN _regex_extract(user_request)
    
    TRY:
        // Create OpenAI client
        client = OpenAI(api_key: api_key)
        model = get_env("OPENAI_MODEL", "gpt-4o-mini")
        
        // Create system prompt
        system = "You extract RF requirements from a short user prompt. " +
                "Output ONLY JSON with keys: freq_low, freq_high, gain_db, noise_figure_db. " +
                "Use GHz units for frequency (floats). If a single frequency is given, set both " +
                "freq_low and freq_high to that value. Missing values must be null."
        
        // Make API call
        response = client.chat.completions.create(
            model: model,
            messages: [
                {role: "system", content: system},
                {role: "user", content: "Prompt: " + user_request}
            ],
            response_format: {type: "json_object"},
            temperature: 0
        )
        
        // Parse response
        content = response.choices[0].message.content OR "{}"
        data = parse_json(content)
        
        // Normalize keys and types
        result = {
            freq_low: data.get("freq_low"),
            freq_high: data.get("freq_high"),
            gain_db: data.get("gain_db"),
            noise_figure_db: data.get("noise_figure_db")
        }
        
        // Ensure numbers if present
        FOR key IN result.keys():
            value = result[key]
            IF value is not null:
                TRY:
                    result[key] = float(value)
                CATCH:
                    result[key] = null
        
        // Fallback sanity check
        IF result.freq_low is null OR result.freq_high is null:
            regex_result = _regex_extract(user_request)
            FOR key, value IN regex_result.items():
                IF result.get(key) is null:
                    result[key] = value
        
        RETURN result
        
    CATCH Exception:
        // Final fallback
        RETURN _regex_extract(user_request)
```

## 7. CHAT MANAGER (chat_manager.py)

### 7.1 Conversation Manager Class
```
CLASS ConversationManager:
    conversations: Dict[string, List[ChatMessage]]
    
    FUNCTION __init__():
        conversations = {}
    
    FUNCTION _get_openai_client():
        // Load API key dynamically
        api_key = get_env("OPENAI_API_KEY")
        IF not api_key OR api_key == "your_openai_api_key_here":
            PRINT "OpenAI API key not found or invalid"
            RETURN null
        
        TRY:
            client = OpenAI(api_key: api_key)
            RETURN client
        CATCH Exception:
            PRINT "Error creating OpenAI client: " + str(e)
            RETURN null
    
    FUNCTION _create_system_prompt():
        RETURN "You are an expert RF component recommendation assistant. " +
               "You help users find the perfect LNA (Low Noise Amplifier) and Bias-T components " +
               "for their RF systems. When users ask for component recommendations: " +
               "1. Extract their requirements (frequency range, gain, noise figure, etc.) " +
               "2. Provide specific component recommendations with part numbers " +
               "3. Explain why these components are suitable " +
               "4. Offer additional advice if needed"
    
    FUNCTION _format_recommendations(lna_result, bias_t_result):
        IF not lna_result OR not bias_t_result:
            RETURN "I couldn't find suitable components with the given specifications. Please try different parameters."
        
        response = "\n\n**Component Recommendations:**\n\n"
        
        // LNA Recommendation
        response += "**📡 LNA (Low Noise Amplifier):**\n"
        response += "• Part Number: " + lna_result.get('part_number', 'N/A') + "\n"
        response += "• Manufacturer: " + lna_result.get('manufacturer', 'N/A') + "\n"
        response += "• Frequency Range: " + lna_result.get('frequency_range', 'N/A') + "\n"
        response += "• Gain: " + lna_result.get('gain', 'N/A') + "\n"
        response += "• Noise Figure: " + lna_result.get('noise_figure', 'N/A') + "\n"
        response += "• Package: " + lna_result.get('package', 'N/A') + "\n"
        
        // Bias-T Recommendation
        response += "\n**⚡ Bias-T:**\n"
        response += "• Part Number: " + bias_t_result.get('part_number', 'N/A') + "\n"
        response += "• Manufacturer: " + bias_t_result.get('manufacturer', 'N/A') + "\n"
        response += "• Frequency Range: " + bias_t_result.get('frequency_range', 'N/A') + "\n"
        response += "• Insertion Loss: " + bias_t_result.get('insertion_loss', 'N/A') + "\n"
        response += "• Max DC Voltage: " + bias_t_result.get('max_dc_voltage', 'N/A') + "\n"
        response += "• Max DC Current: " + bias_t_result.get('max_dc_current', 'N/A') + "\n"
        
        response += "\nThese components should work well together for your RF system requirements."
        RETURN response
    
    FUNCTION _get_component_recommendations(extracted_data):
        TRY:
            // Prepare LNA input
            lna_input = {
                freq_low: extracted_data.get('freq_low', 0),
                freq_high: extracted_data.get('freq_high', 0),
                gain_db: extracted_data.get('gain_db'),
                noise_figure_db: extracted_data.get('noise_figure_db')
            }
            
            // Prepare Bias-T input
            bias_t_input = {
                freq_low: extracted_data.get('freq_low', 0),
                freq_high: extracted_data.get('freq_high', 0)
            }
            
            // Get recommendations
            lna_result = predict_lna(lna_input)
            bias_t_results = recommend_bias_t(bias_t_input, top_k: 1)
            bias_t_result = bias_t_results[0] IF bias_t_results ELSE null
            
            RETURN lna_result, bias_t_result
            
        CATCH Exception:
            PRINT "Error getting recommendations: " + str(e)
            RETURN null, null
    
    FUNCTION process_message(request: ChatRequest):
        // Get or create conversation
        conversation_id = request.conversation_id OR generate_uuid()
        IF conversation_id not in conversations:
            conversations[conversation_id] = []
        
        // Add user message to conversation
        user_message = ChatMessage(role: "user", content: request.message)
        conversations[conversation_id].append(user_message)
        
        // Get OpenAI client
        client = _get_openai_client()
        
        IF not client:
            // Fallback response without OpenAI
            response = "I'm currently unable to access advanced AI features. Please try again later or contact support."
            assistant_message = ChatMessage(role: "assistant", content: response)
            conversations[conversation_id].append(assistant_message)
            RETURN ChatResponse(response: response, conversation_id: conversation_id)
        
        TRY:
            // Prepare conversation history for OpenAI
            messages = [{role: "system", content: _create_system_prompt()}]
            
            // Add conversation history (last 10 messages to avoid token limits)
            recent_messages = conversations[conversation_id][-10:]
            FOR msg IN recent_messages:
                messages.append({role: msg.role, content: msg.content})
            
            // Get OpenAI response
            model = get_env("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model: model,
                messages: messages,
                temperature: 0.7,
                max_tokens: 1000
            )
            
            assistant_response = response.choices[0].message.content
            
            // Check if user is asking for component recommendations
            IF contains_keywords(request.message.lower(), ['recommend', 'lna', 'bias-t', 'bias t', 'amplifier', 'component']):
                // Extract requirements
                extracted_data = extract_requirements_via_openai(request.message)
                
                IF extracted_data.get('freq_low') AND extracted_data.get('freq_high'):
                    // Get component recommendations
                    lna_result, bias_t_result = _get_component_recommendations(extracted_data)
                    
                    IF lna_result AND bias_t_result:
                        // Add recommendations to response
                        recommendations_text = _format_recommendations(lna_result, bias_t_result)
                        assistant_response += recommendations_text
                        
                        // Store recommendations data
                        recommendations_data = {
                            lna: lna_result,
                            bias_t: bias_t_result
                        }
                    ELSE:
                        assistant_response += "\n\nI couldn't find suitable components with the given specifications. Please try different parameters."
                        recommendations_data = null
                ELSE:
                    assistant_response += "\n\nI need more specific information about your requirements. Please specify frequency range, gain, noise figure, etc."
                    recommendations_data = null
            ELSE:
                recommendations_data = null
            
            // Add assistant message to conversation
            assistant_message = ChatMessage(role: "assistant", content: assistant_response)
            conversations[conversation_id].append(assistant_message)
            
            RETURN ChatResponse(
                response: assistant_response,
                conversation_id: conversation_id,
                recommendations: recommendations_data
            )
            
        CATCH Exception:
            error_response = "I encountered an error: " + str(e) + ". Please try again."
            assistant_message = ChatMessage(role: "assistant", content: error_response)
            conversations[conversation_id].append(assistant_message)
            RETURN ChatResponse(response: error_response, conversation_id: conversation_id)
    
    FUNCTION get_conversation_history(conversation_id: string):
        RETURN conversations.get(conversation_id, [])
    
    FUNCTION clear_conversation(conversation_id: string):
        IF conversation_id in conversations:
            DELETE conversations[conversation_id]
            RETURN true
        RETURN false

// Global conversation manager instance
conversation_manager = ConversationManager()
```

## 8. WEB INTERFACE (static/index.html)

### 8.1 Frontend Structure
```
HTML Structure:
├── Chat Container
│   ├── Chat Header
│   │   ├── Title and Description
│   │   ├── Status Indicator
│   │   └── Clear Chat Button
│   ├── Chat Messages Area
│   └── Chat Input Area
│       ├── Input Field
│       ├── Send Button
│       └── Example Queries
```

### 8.2 JavaScript Functions
```
VARIABLES:
- apiBaseUrl: window.location.origin
- isProcessing: boolean
- conversationId: string or null

FUNCTION setStatus(message, type):
    statusText.textContent = message
    statusIndicator.className = "status-indicator " + type

FUNCTION setExampleQuery(query):
    userInput.value = query
    userInput.focus()

FUNCTION handleKeyPress(event):
    IF event.key === 'Enter' AND not isProcessing:
        sendMessage()

FUNCTION addMessage(content, isUser = false):
    messageDiv = create_element('div')
    messageDiv.className = "message " + (isUser ? 'user' : 'bot')
    
    contentDiv = create_element('div')
    contentDiv.className = 'message-content'
    
    IF isUser:
        contentDiv.textContent = content
    ELSE:
        // Convert markdown-like formatting to HTML
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>')
        content = content.replace(/\n/g, '<br>')
        contentDiv.innerHTML = content
    
    messageDiv.appendChild(contentDiv)
    chatMessages.appendChild(messageDiv)
    chatMessages.scrollTop = chatMessages.scrollHeight

FUNCTION setLoading(loading):
    isProcessing = loading
    IF loading:
        sendButton.disabled = true
        buttonText.style.display = 'none'
        loadingSpinner.style.display = 'inline-block'
        userInput.disabled = true
    ELSE:
        sendButton.disabled = false
        buttonText.style.display = 'inline'
        loadingSpinner.style.display = 'none'
        userInput.disabled = false
        userInput.focus()

FUNCTION sendMessage():
    message = userInput.value.trim()
    
    IF not message OR isProcessing:
        RETURN
    
    // Add user message
    addMessage(message, true)
    userInput.value = ''
    
    setLoading(true)
    
    TRY:
        // Send message to chat API
        response = fetch(apiBaseUrl + "/chat", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId
            })
        })
        
        IF not response.ok:
            THROW Error("Chat API error: " + response.status)
        
        result = response.json()
        
        // Store conversation ID for future messages
        conversationId = result.conversation_id
        
        // Add bot response
        addMessage(result.response)
        
    CATCH error:
        console.error('Error:', error)
        addMessage("❌ Error: " + error.message + ". Please try again.")
    FINALLY:
        setLoading(false)

FUNCTION clearChat():
    IF conversationId:
        TRY:
            fetch(apiBaseUrl + "/conversation/" + conversationId, {
                method: 'DELETE'
            })
        CATCH error:
            console.error('Error clearing conversation:', error)
    
    // Clear UI
    chatMessages.innerHTML = create_welcome_message()
    conversationId = null

FUNCTION window.onload():
    TRY:
        response = fetch(apiBaseUrl + "/health")
        IF response.ok:
            setStatus('Connected to API ✅', 'connected')
        ELSE:
            setStatus('API connection issue', 'error')
    CATCH error:
        setStatus('Cannot connect to API', 'error')
```

## 9. SYSTEM FLOW

### 9.1 Complete User Interaction Flow
```
1. USER INTERACTION:
   User opens web interface → Frontend loads → Health check → Display status

2. CHAT INITIATION:
   User types message → Frontend sends POST /chat → Backend processes

3. MESSAGE PROCESSING:
   ChatManager receives message → Check for OpenAI client → Prepare conversation history

4. REQUIREMENT EXTRACTION:
   IF message contains recommendation keywords:
       Extract requirements via OpenAI (fallback to regex) → Parse frequency, gain, noise figure

5. COMPONENT RECOMMENDATION:
   IF requirements extracted successfully:
       Prepare LNA input → Call predict_lna() → Prepare Bias-T input → Call recommend_bias_t()
       Format recommendations → Add to response

6. AI RESPONSE:
   Send conversation to OpenAI → Get AI response → Combine with recommendations → Return to user

7. FRONTEND UPDATE:
   Receive response → Display message → Update conversation history → Enable input
```

### 9.2 Model Training Flow
```
1. DATA PREPARATION:
   Load Excel files → Clean and preprocess data → Handle missing values

2. FEATURE ENGINEERING:
   Split frequency ranges → Convert units → Encode categorical variables → Scale numeric features

3. MODEL TRAINING:
   Train NearestNeighbors models → Save model artifacts → Save metadata

4. MODEL PERSISTENCE:
   Save trained models → Save encoders → Save feature columns → Save component data
```

### 9.3 Recommendation Flow
```
1. INPUT VALIDATION:
   Validate input parameters → Convert to required format → Handle missing values

2. FEATURE PREPARATION:
   Load saved models and data → Prepare input features → Apply preprocessing

3. MODEL PREDICTION:
   Run NearestNeighbors search → Get top matches → Extract component information

4. RESPONSE FORMATTING:
   Format component specifications → Add datasheet URLs → Return structured response
```

## 10. ERROR HANDLING

### 10.1 Graceful Degradation
```
- OpenAI API unavailable → Fallback to regex extraction
- Model loading fails → Return error with helpful message
- Invalid input → Validate and provide guidance
- Network issues → Retry with exponential backoff
```

### 10.2 Data Validation
```
- Frequency ranges → Ensure positive values, convert units
- Numeric fields → Handle NaN, infinite values
- Categorical fields → Handle unknown categories
- API responses → Validate JSON structure
```

This pseudocode provides a comprehensive overview of the entire RF Component Recommendation Service, covering all major components, data flows, and error handling mechanisms.
