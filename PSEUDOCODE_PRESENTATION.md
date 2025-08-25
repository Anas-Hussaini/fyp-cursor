# RF Component Recommendation Service - Presentation Slides

---

## Slide 1: Title Slide
# RF Component Recommendation Service
## System Architecture & Pseudocode Overview

**Presented by:** [Your Name]  
**Date:** [Presentation Date]

---

## Slide 2: System Overview
# System Architecture

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

**Key Components:**
- **Backend API**: FastAPI server with REST endpoints
- **ML Models**: Nearest Neighbors for component matching
- **NLP**: OpenAI + regex for requirement extraction
- **Chat System**: Conversational AI with session management
- **Frontend**: Modern web interface

---

## Slide 3: Main Application Flow
# FastAPI Application (main.py)

## Application Initialization
```pseudocode
FUNCTION initialize_application():
    load_dotenv()                    // Load environment variables
    app = FastAPI(                   // Create FastAPI app
        title: "RF Component Recommendation API",
        description: "API for recommending LNA and Bias-T components",
        version: "1.0.0"
    )
    app.add_middleware(CORS)         // Enable CORS
    app.mount("/static", StaticFiles) // Serve web interface
    RETURN app
```

## Key Endpoints
- `GET /` - API information
- `POST /chat` - Conversational AI
- `POST /recommend-lna` - LNA recommendations
- `POST /recommend-bias-t` - Bias-T recommendations
- `GET /health` - Health check

---

## Slide 4: Data Models
# Pydantic Data Models (models.py)

## Input Models
```pseudocode
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

## Response Models
```pseudocode
CLASS LNARecommendation:
    part_number: string
    manufacturer: string
    frequency_range: string
    gain: string
    noise_figure: string
    package: string
    datasheet_url: string

CLASS ChatResponse:
    response: string
    conversation_id: string
    recommendations: optional dict
```

---

## Slide 5: LNA Model Training
# LNA Recommendation Model (lna_model.py)

## Training Process
```pseudocode
FUNCTION train_lna_model(excel_file_path):
    // 1. Load and preprocess data
    df = read_excel(excel_file_path)
    
    // 2. Clean frequency ranges (GHz to MHz)
    FOR each row IN df:
        freq_range = row['Freq Range (GHz)']
        IF freq_range is not null:
            low, high = split_frequency_range(freq_range)
            row['Freq Low (MHz)'] = low * 1000
            row['Freq High (MHz)'] = high * 1000
        
        // Clean numeric fields
        row['Gain (dB)'] = to_numeric(row['Gain (dB)'])
        row['Noise Figure (dB)'] = to_numeric(row['Noise Figure (dB)'])
    
    // 3. Feature selection
    drop_columns = ['Part Number', 'Manufacturer', 'Datasheet Link', 'Freq Range (GHz)']
    X = df.drop(columns: drop_columns)
    
    // 4. Train Nearest Neighbors model
    nn = NearestNeighbors(n_neighbors: 1)
    nn.fit(X)
    
    // 5. Save artifacts
    save_model(nn, 'models/lna_nn_model.joblib')
    save_data(part_numbers, 'models/lna_part_numbers.joblib')
    // ... save other metadata
```

---

## Slide 6: LNA Prediction
# LNA Component Prediction

## Prediction Algorithm
```pseudocode
FUNCTION predict_lna(input_features):
    // 1. Load trained model and data
    lna_nn = load_model('models/lna_nn_model.joblib')
    part_numbers = load_data('models/lna_part_numbers.joblib')
    manufacturers = load_data('models/lna_manufacturers.joblib')
    // ... load other data
    
    // 2. Prepare input features
    input_df = create_dataframe([input_features])
    input_df = ensure_columns_match_training(input_df)
    
    // 3. Find nearest neighbor
    distances, indices = lna_nn.kneighbors(input_df, n_neighbors: 1)
    idx = indices[0][0]
    
    // 4. Return formatted recommendation
    RETURN {
        part_number: part_numbers[idx],
        manufacturer: manufacturers[idx],
        frequency_range: freq_ranges[idx],
        gain: format_gain(gains[idx]),
        noise_figure: format_noise_figure(noise_figures[idx]),
        package: connector_types[idx],
        datasheet_url: datasheet_urls[idx]
    }
```

---

## Slide 7: Bias-T Model Training
# Bias-T Recommendation Model (bias_t_model.py)

## Advanced Training Process
```pseudocode
FUNCTION train_bias_t_model(excel_file_path):
    // 1. Load and preprocess data
    df = read_excel(excel_file_path)
    
    // 2. Handle categorical variables
    categorical_columns = ['Connector Type', 'Manufacturer']
    numeric_columns = ['Freq Low (MHz)', 'Freq High (MHz)', 'Insertion Loss (dB)',
                      'Return Loss (dB)', 'Max DC Voltage (V)', 'Max DC Current (mA)']
    
    // 3. Encode categorical variables
    label_encoders = {}
    FOR column IN categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].fill_na('Unknown'))
        label_encoders[column] = le
    
    // 4. Scale numeric variables
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    // 5. Train model
    nn = NearestNeighbors(n_neighbors: 5)
    nn.fit(df.drop(columns: ['Part Number', 'Datasheet Link', 'Freq Range (GHz)']))
    
    // 6. Save all artifacts
    save_model(nn, 'models/bias_t_nn_model.joblib')
    save_model(scaler, 'models/bias_t_scaler.joblib')
    FOR column IN categorical_columns:
        save_model(label_encoders[column], f'models/bias_t_le_{column}.joblib')
```

---

## Slide 8: Bias-T Recommendation
# Bias-T Component Recommendation

## Recommendation Algorithm
```pseudocode
FUNCTION recommend_bias_t(input_features, top_k: 1):
    // 1. Load models and encoders
    bias_nn = load_model('models/bias_t_nn_model.joblib')
    bias_scaler = load_model('models/bias_t_scaler.joblib')
    le_connector = load_model('models/bias_t_le_Connector Type.joblib')
    le_manufacturer = load_model('models/bias_t_le_Manufacturer.joblib')
    
    // 2. Prepare input with preprocessing
    input_df = create_dataframe([{
        'Freq Low (MHz)': input_features.get('freq_low', 0) * 1000,
        'Freq High (MHz)': input_features.get('freq_high', 0) * 1000,
        'Connector Type': 'Unknown',
        'Manufacturer': 'Unknown'
        // ... other default values
    }])
    
    // 3. Apply same preprocessing as training
    input_df['Connector Type'] = le_connector.transform(input_df['Connector Type'])
    input_df['Manufacturer'] = le_manufacturer.transform(input_df['Manufacturer'])
    input_df[numeric_columns] = bias_scaler.transform(input_df[numeric_columns])
    
    // 4. Find top-k matches
    distances, indices = bias_nn.kneighbors(input_df, n_neighbors: top_k)
    
    // 5. Format recommendations
    recommendations = []
    FOR idx IN indices[0]:
        recommendations.append(format_component_data(idx))
    
    RETURN recommendations
```

---

## Slide 9: Natural Language Processing
# NLP Requirement Extraction (nlp_extractor.py)

## Dual Extraction Strategy
```pseudocode
FUNCTION extract_requirements_via_openai(user_request: string):
    // 1. Check for OpenAI API availability
    api_key = get_env("OPENAI_API_KEY")
    IF not api_key:
        RETURN _regex_extract(user_request)  // Fallback
    
    // 2. Use OpenAI for intelligent extraction
    TRY:
        client = OpenAI(api_key: api_key)
        response = client.chat.completions.create(
            model: "gpt-4o-mini",
            messages: [
                {role: "system", content: "Extract RF requirements. Output JSON with keys: freq_low, freq_high, gain_db, noise_figure_db"},
                {role: "user", content: "Prompt: " + user_request}
            ],
            response_format: {type: "json_object"},
            temperature: 0
        )
        
        // 3. Parse and validate response
        data = parse_json(response.choices[0].message.content)
        result = normalize_and_validate(data)
        
        // 4. Fallback to regex if needed
        IF result.freq_low is null OR result.freq_high is null:
            regex_result = _regex_extract(user_request)
            result = merge_results(result, regex_result)
        
        RETURN result
        
    CATCH Exception:
        RETURN _regex_extract(user_request)  // Final fallback
```

---

## Slide 10: Regex Fallback Extraction
# Regex Pattern Matching

## Pattern-Based Extraction
```pseudocode
FUNCTION _regex_extract(user_request: string):
    extracted = {
        freq_low: null,
        freq_high: null,
        gain_db: null,
        noise_figure_db: null
    }
    
    // Extract frequency range patterns
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
    
    // Extract single frequency
    single_match = search_regex(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)", user_request)
    IF single_match:
        value = float(single_match.group(1))
        unit = single_match.group(2).lower()
        IF unit == "mhz":
            value /= 1000.0
        extracted.freq_low = value
        extracted.freq_high = value
    
    // Extract gain and noise figure
    gain_match = search_regex(r"(\d+\.?\d*)\s*dB.*gain", user_request)
    IF gain_match:
        extracted.gain_db = float(gain_match.group(1))
    
    noise_match = search_regex(r"(\d+\.?\d*)\s*dB.*noise", user_request)
    IF noise_match:
        extracted.noise_figure_db = float(noise_match.group(1))
    
    RETURN extracted
```

---

## Slide 11: Conversational AI Manager
# Chat Manager (chat_manager.py)

## Conversation Management
```pseudocode
CLASS ConversationManager:
    conversations: Dict[string, List[ChatMessage]]
    
    FUNCTION process_message(request: ChatRequest):
        // 1. Manage conversation session
        conversation_id = request.conversation_id OR generate_uuid()
        IF conversation_id not in conversations:
            conversations[conversation_id] = []
        
        // 2. Add user message
        user_message = ChatMessage(role: "user", content: request.message)
        conversations[conversation_id].append(user_message)
        
        // 3. Get OpenAI client
        client = _get_openai_client()
        IF not client:
            RETURN fallback_response()
        
        // 4. Prepare conversation history
        messages = [{role: "system", content: _create_system_prompt()}]
        recent_messages = conversations[conversation_id][-10:]  // Last 10 messages
        FOR msg IN recent_messages:
            messages.append({role: msg.role, content: msg.content})
        
        // 5. Get AI response
        response = client.chat.completions.create(
            model: "gpt-4o-mini",
            messages: messages,
            temperature: 0.7,
            max_tokens: 1000
        )
        
        assistant_response = response.choices[0].message.content
```

---

## Slide 12: Component Recommendation Integration
# AI + ML Integration

## Smart Recommendation Flow
```pseudocode
FUNCTION process_message(request: ChatRequest):
    // ... previous code ...
    
    // Check if user wants component recommendations
    IF contains_keywords(request.message.lower(), ['recommend', 'lna', 'bias-t', 'amplifier']):
        // 1. Extract requirements using NLP
        extracted_data = extract_requirements_via_openai(request.message)
        
        // 2. Validate extracted data
        IF extracted_data.get('freq_low') AND extracted_data.get('freq_high'):
            // 3. Get ML recommendations
            lna_result, bias_t_result = _get_component_recommendations(extracted_data)
            
            // 4. Format and add to response
            IF lna_result AND bias_t_result:
                recommendations_text = _format_recommendations(lna_result, bias_t_result)
                assistant_response += recommendations_text
                
                recommendations_data = {
                    'lna': lna_result,
                    'bias_t': bias_t_result
                }
            ELSE:
                assistant_response += "\n\nI couldn't find suitable components."
        ELSE:
            assistant_response += "\n\nI need more specific information about your requirements."
    
    // 5. Return combined response
    RETURN ChatResponse(
        response: assistant_response,
        conversation_id: conversation_id,
        recommendations: recommendations_data
    )
```

---

## Slide 13: Web Interface
# Frontend Implementation (static/index.html)

## Modern Chat Interface
```pseudocode
// Frontend JavaScript Functions
VARIABLES:
- apiBaseUrl: window.location.origin
- isProcessing: boolean
- conversationId: string or null

FUNCTION sendMessage():
    message = userInput.value.trim()
    
    IF not message OR isProcessing:
        RETURN
    
    // 1. Add user message to UI
    addMessage(message, true)
    userInput.value = ''
    setLoading(true)
    
    // 2. Send to backend API
    TRY:
        response = fetch(apiBaseUrl + "/chat", {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId
            })
        })
        
        // 3. Handle response
        IF not response.ok:
            THROW Error("Chat API error: " + response.status)
        
        result = response.json()
        conversationId = result.conversation_id
        
        // 4. Display bot response with formatting
        addMessage(result.response)
        
    CATCH error:
        addMessage("❌ Error: " + error.message + ". Please try again.")
    FINALLY:
        setLoading(false)
```

---

## Slide 14: Complete System Flow
# End-to-End User Interaction

## User Journey
```pseudocode
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

---

## Slide 15: Model Training Flow
# Machine Learning Pipeline

## Training Process
```pseudocode
1. DATA PREPARATION:
   Load Excel files → Clean and preprocess data → Handle missing values

2. FEATURE ENGINEERING:
   Split frequency ranges → Convert units → Encode categorical variables → Scale numeric features

3. MODEL TRAINING:
   Train NearestNeighbors models → Save model artifacts → Save metadata

4. MODEL PERSISTENCE:
   Save trained models → Save encoders → Save feature columns → Save component data
```

---

## Slide 16: Recommendation Flow
# ML Prediction Pipeline

## Prediction Process
```pseudocode
1. INPUT VALIDATION:
   Validate input parameters → Convert to required format → Handle missing values

2. FEATURE PREPARATION:
   Load saved models and data → Prepare input features → Apply preprocessing

3. MODEL PREDICTION:
   Run NearestNeighbors search → Get top matches → Extract component information

4. RESPONSE FORMATTING:
   Format component specifications → Add datasheet URLs → Return structured response
```

---

## Slide 17: Error Handling Strategy
# Graceful Degradation

## Robust Error Handling
```pseudocode
// Error Handling Strategy
1. OPENAI API UNAVAILABLE:
   - Fallback to regex extraction
   - Provide helpful error message
   - Continue with basic functionality

2. MODEL LOADING FAILS:
   - Return error with helpful message
   - Suggest retraining or check model files
   - Log detailed error for debugging

3. INVALID INPUT:
   - Validate input parameters
   - Provide guidance on required format
   - Use default values where appropriate

4. NETWORK ISSUES:
   - Retry with exponential backoff
   - Show connection status to user
   - Cache responses where possible
```

---

## Slide 18: Data Validation
# Input Validation & Sanitization

## Comprehensive Validation
```pseudocode
// Data Validation
1. FREQUENCY RANGES:
   - Ensure positive values
   - Convert units (MHz to GHz)
   - Handle range formats (1-6 GHz, 1 to 6 GHz)

2. NUMERIC FIELDS:
   - Handle NaN and infinite values
   - Validate ranges (gain > 0, noise figure > 0)
   - Convert string representations

3. CATEGORICAL FIELDS:
   - Handle unknown categories
   - Normalize manufacturer names
   - Validate connector types

4. API RESPONSES:
   - Validate JSON structure
   - Check for required fields
   - Handle malformed responses
```

---

## Slide 19: Key Features Summary
# System Highlights

## ✨ Key Features

### 🤖 **Intelligent NLP**
- OpenAI-powered requirement extraction
- Regex fallback for reliability
- Natural language understanding

### 🧠 **Machine Learning**
- Nearest Neighbors for component matching
- Pre-trained models for LNA and Bias-T
- Real-time recommendations

### 💬 **Conversational AI**
- Context-aware chat sessions
- Component recommendation integration
- Helpful explanations and guidance

### 🌐 **Modern Web Interface**
- Responsive design
- Real-time chat experience
- Example queries for easy testing

### 🔧 **Robust Architecture**
- Graceful error handling
- Multiple fallback mechanisms
- Comprehensive data validation

---

## Slide 20: Technical Stack
# Technology Overview

## Backend Technologies
- **FastAPI**: High-performance web framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **OpenAI API**: Natural language processing
- **Pydantic**: Data validation and serialization

## Frontend Technologies
- **HTML5/CSS3**: Modern web interface
- **JavaScript (ES6+)**: Interactive functionality
- **Fetch API**: Asynchronous communication

## Deployment & Infrastructure
- **Uvicorn**: ASGI server
- **CORS**: Cross-origin resource sharing
- **Static file serving**: Web interface hosting
- **Environment variables**: Configuration management

---

## Slide 21: Future Enhancements
# Roadmap & Improvements

## Potential Enhancements

### 🔮 **Advanced Features**
- Multi-component recommendations
- Cost optimization algorithms
- Performance comparison tools
- Integration with supplier APIs

### 📊 **Analytics & Insights**
- Usage analytics dashboard
- Recommendation accuracy metrics
- User feedback collection
- Performance monitoring

### 🔧 **Technical Improvements**
- Database integration for persistence
- Caching mechanisms for performance
- Microservices architecture
- Containerization with Docker

### 🌍 **Scalability**
- Load balancing
- Horizontal scaling
- CDN integration
- Global deployment

---

## Slide 22: Thank You
# Questions & Discussion

## Thank You!

**RF Component Recommendation Service**  
*Intelligent AI-powered component selection for RF systems*

### Contact Information
- **Email**: [your.email@domain.com]
- **GitHub**: [github.com/yourusername]
- **LinkedIn**: [linkedin.com/in/yourprofile]

### Resources
- **Documentation**: `/docs` endpoint
- **API Reference**: `/docs` (Swagger UI)
- **Web Interface**: `/interface` endpoint
- **Health Check**: `/health` endpoint

**Questions?** 🤔

---

*End of Presentation*
