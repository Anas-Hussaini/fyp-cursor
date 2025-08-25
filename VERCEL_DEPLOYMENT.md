# Vercel Deployment Guide

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **OpenAI API Key**: You'll need an OpenAI API key for the chat and NLP features

## Environment Variables

Before deploying, you need to set up these environment variables in Vercel:

### Required Environment Variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4o-mini)

### Optional Environment Variables:
- `PORT`: Server port (Vercel sets this automatically)
- `HOST`: Server host (Vercel sets this automatically)

## Deployment Steps

### Method 1: Deploy via Vercel Dashboard

1. **Connect Repository**:
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository: `Anas-Hussaini/fyp-cursor`

2. **Configure Project**:
   - Framework Preset: Other
   - Root Directory: `./` (leave as default)
   - Build Command: Leave empty (Vercel will auto-detect)
   - Output Directory: Leave empty

3. **Set Environment Variables**:
   - In the project settings, go to "Environment Variables"
   - Add `OPENAI_API_KEY` with your actual API key
   - Add `OPENAI_MODEL` with value `gpt-4o-mini`

4. **Deploy**:
   - Click "Deploy"
   - Wait for the build to complete

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Set Environment Variables**:
   ```bash
   vercel env add OPENAI_API_KEY
   vercel env add OPENAI_MODEL
   ```

## Project Structure for Vercel

The deployment uses these key files:

- `vercel.json`: Vercel configuration
- `api/index.py`: Serverless function entry point
- `main.py`: Main FastAPI application
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python version specification

## API Endpoints

Once deployed, your API will be available at:

- **Base URL**: `https://your-project-name.vercel.app`
- **Health Check**: `GET /health`
- **API Documentation**: `GET /docs`
- **Web Interface**: `GET /interface`
- **Chat**: `POST /chat`
- **LNA Recommendation**: `POST /recommend-lna`
- **Bias-T Recommendation**: `POST /recommend-bias-t`
- **Extract Requirements**: `POST /extract-requirements`

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version in `runtime.txt` is supported

2. **Environment Variables**:
   - Make sure `OPENAI_API_KEY` is set correctly
   - Check Vercel dashboard for environment variable configuration

3. **Import Errors**:
   - The `api/index.py` file handles Python path issues
   - All imports should work correctly

4. **Function Timeout**:
   - Vercel has a 30-second timeout for serverless functions
   - Complex ML operations might need optimization

### Support:

- Check Vercel logs in the dashboard
- Review build output for specific error messages
- Ensure all files are committed to GitHub before deployment
