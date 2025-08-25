# RF Component Recommendation Service - Deployment Guide

## Vercel Deployment

This service has been configured for deployment on Vercel with the following structure:

### Files Structure
```
├── main.py                 # FastAPI application
├── models.py               # Pydantic models
├── chat_manager.py         # Conversation management
├── nlp_extractor.py        # OpenAI integration
├── lna_model.py           # LNA recommendation logic
├── bias_t_model.py        # Bias-T recommendation logic
├── static/                # Web interface files
│   └── index.html
├── models/                # Trained ML models
├── vercel.json            # Vercel configuration
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

### Environment Variables

Set these environment variables in your Vercel project:

1. **OPENAI_API_KEY**: Your OpenAI API key
2. **OPENAI_MODEL**: Model to use (default: gpt-4o-mini)

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Set environment variables
   - Deploy

3. **Environment Variables in Vercel**:
   - Go to your project settings
   - Add environment variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `OPENAI_MODEL`: gpt-4o-mini (optional)

### API Endpoints

After deployment, your API will be available at:
- **Base URL**: `https://your-project.vercel.app`
- **Health Check**: `GET /health`
- **Chat API**: `POST /chat`
- **LNA Recommendations**: `POST /recommend-lna`
- **Bias-T Recommendations**: `POST /recommend-bias-t`
- **Web Interface**: `GET /interface`
- **API Documentation**: `GET /docs`

### Important Notes

1. **Model Files**: Ensure your trained models are in the `models/` directory
2. **Environment Variables**: Don't commit sensitive data like API keys
3. **CORS**: The API is configured to allow all origins for development
4. **Function Timeout**: Set to 30 seconds for Vercel serverless functions

### Troubleshooting

- **Cold Start**: First request may be slow due to model loading
- **Memory Limits**: Vercel has memory limits for serverless functions
- **Timeout**: Complex requests may timeout at 30 seconds

### Local Development

For local development, run:
```bash
pip install -r requirements.txt
python main.py
```

The service will be available at `http://localhost:8000`
