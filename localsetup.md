# GREENPETAL — Complete Local Setup Guide

This guide walks you through every step needed to run GREENPETAL locally, including the backend API, frontend web interface, and all ML models.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Project Structure Overview](#project-structure-overview)
4. [Backend Setup](#backend-setup)
5. [Frontend Setup](#frontend-setup)
6. [Model Training (Optional)](#model-training-optional)
7. [Running the Full Application](#running-the-full-application)
8. [API Endpoints Reference](#api-endpoints-reference)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8 GB minimum (16 GB recommended for model training)
- **Disk Space**: ~4-5 GB for dependencies and model weights
- **Internet**: Required for initial setup and model data downloads

---

## Prerequisites Installation

### 1. Python 3.11

1. Download Python 3.11 from https://www.python.org/downloads/
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```powershell
   python --version
   # Output: Python 3.11.x
   ```

### 2. Node.js 20 and npm

1. Download Node.js 20 LTS from https://nodejs.org/
2. Run the installer and follow the prompts
3. Verify installation:
   ```powershell
   node --version
   # Output: v20.x.x
   npm --version
   # Output: 10.x.x
   ```

### 3. MongoDB

MongoDB is required for storing analysis results and job history.

**Windows:**
1. Download MongoDB Community from https://www.mongodb.com/try/download/community
2. Run the installer and accept default settings
3. MongoDB runs as a Windows Service by default
4. Verify it's running:
   ```powershell
   # Test connection
   python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print('Connected:', client.admin.command('ping'))"
   # Output: Connected: {'ok': 1.0}
   ```

**macOS:**
1. Download MongoDB Community from https://www.mongodb.com/try/download/community
2. Run the installer and accept default settings
3. MongoDB runs as a Windows Service by default
4. Verify it's running:
   ```powershell
   # Test connection
   python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print('Connected:', client.admin.command('ping'))"
   # Output: Connected: {'ok': 1.0}
   ├── backend/                      # FastAPI server
│   ├── avle/                    # Main ML pipeline
│   │   ├── train_all.py         # Orchestrator for training all models
│   │   ├── train_carbon.py      # Carbon model training
│   │   ├── train_prediction.py  # Prediction model training
│   │   ├── train_recommendation.py  # Recommendation model training
│   │   ├── train_segmentation_real.py  # Segmentation model training
│   │   ├── pipeline.py          # Main analysis pipeline
│   │   ├── blockchain.py        # Blockchain integration
│   │   ├── config.py            # Configuration
│   │   ├── evaluate.py          # Model evaluation
│   │   ├── weights/             # Trained model weights (auto-populated)
│   │   └── data/                # Data cache
│   ├── server.py                # FastAPI entry point
│   ├── requirements.txt          # Python dependencies
│   ├── start_backend.ps1         # Backend startup script
│   └── .env                      # Environment variables (if needed)
├── frontend/                     # React web interface
│   ├── src/
│   │   ├── components/          # UI components
│   │   ├── pages/               # Page components
│   │   ├── hooks/               # Custom React hooks
│   │   ├── lib/                 # Utility functions
│   │   └── App.js               # Main app component
│   ├── package.json             # Node dependencies
│   ├── public/                  # Static assets
│   └── .env                      # Frontend environment variables
├── docs/                        # Documentation
│   ├── SETUP.md                # Original setup guide
│   ├── MODEL_TRAINING_GUIDE.md # Model training details
│   └── RESEARCH_PAPER_GUIDE.md # API documentation
└── localsetup.md               # This file
```

---

## Backend Setup

### Step 1: Navigate to Backend Directory

**Windows (PowerShell):**
```powershell
cd C:\Users\YourUsername\Downloads\GREENPETAL-main\backend
```

**macOS/Linux:**
```bash
cd ~/Downloads/GREENPETAL-main/backend
```

### Step 2: Create Python Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Note: If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You should see `(venv)` prefix in your PowerShell

# Install requirements
pip install -r requirements.txt
```

This will take 5-10 minutes. The installation includes:
- **FastAPI** - Web framework
- **uvicorn** - ASGI server
- **motor** - Async MongoDB driver
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **torch** - Deep learning (CPU or CUDA)
- **rasterio** - Geospatial raster data
- **Sentinel-2** and **ESA WorldCover** tools
- **web3** - Blockchain integration

### Step 4: Environment Configuration

Create a `.env` file in the `backend/` directory:

```bash
# backend/.env
MONGO_URL=mongodb://localhost:27017
DB_NAME=avle
CORS_ORIGINS=*
```

Or if you already have one, verify these variables are set.

### Step 5: Test Backend Server

```bash
# Run the backend server
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Test the health endpoint:**
```powershell
# In a new terminal
curl http://127.0.0.1:8000/api/
# Output: {"service":"AVLE-C","version":"1.0.0","status":"ok"}
```

Press `Ctrl+C` to stop the server when done testing.

---

## Frontend Setup

### Step 1: Navigate to Frontend Directory

**Windows (PowerShell):**
```powershell
cd C:\Users\YourUsername\Downloads\GREENPETAL-main\frontend
```

**macOS/Linux:**
```bash
```powershell
cd C:\Users\YourUsername\Downloads\GREENPETAL-main\

This will take 3-5 minutes and install all React components, UI libraries, and utilities including:
- **React 19** - UI framework
- **React Router** - Navigation
- **Recharts** - Data visualization
- **Leaflet** - Map visualization
- **Tailwind CSS** - Styling
- **Radix UI** - Component library

### Step 3: Environment Configuration

Create a `.env` file in the `frontend/` directory:

```bash
# frontend/.env
REACT_APP_BACKEND_URL=http://127.0.0.1:8000
```

This tells the frontend where the backend API is running.

### Step 4: Test Frontend Server

```bash
npm start
```

**Expected output:**
```
Compiled successfully!
You can now view frontend in the browser.
Local: http://localhost:3000
```

The app will automatically open in your default browser. If not, visit `http://localhost:3000`.

Press `Ctrl+C` to stop when done testing.

---

## Model Training (Optional)

### Understanding the Models

GREENPETAL uses several ML models for different tasks:

1. **U-Net (Segmentation)** - Land cover segmentation from Sentinel-2 imagery
2. **Carbon Model** - Estimates carbon emissions/sequestration
3. **Recommendation Model** - Suggests mitigation strategies
4. **Prediction Model** - Forecasts future outcomes
5. **XGBoost Models** - Business-as-usual, mitigation, and uncertainty quantile models
6. **ARIMA Baseline** - Time series baseline for comparison

### What Training Does

- Downloads ~16 Sentinel-2 and ESA WorldCover tiles (cached locally)
- Trains U-Net on satellite imagery
- Trains forest and gradient boosting models
- Evaluates all models
- Generates ablation table (Table 1 in research paper)
- Saves weights to `backend/avle/weights/`

### How Much Storage/Time?

- **Download**: ~2-3 GB (cached to `backend/avle/data/seg_cache/`)
- **Time**: 4-6 minutes on CPU, 1-2 minutes on GPU
- **Disk for weights**: ~200 MB

### Run Model Training

**Ensure backend virtual environment is activated:**

```bash
# If you closed the terminal, reactivate:
cd backend
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate     # macOS/Linux
```

**Train all models:**

```bash
python -m avle.train_all
```
powershell
# If you closed the terminal, reactivate:
cd backend
.\venv\Scripts\Activate.ps1

# Carbon model only
python -m avle.train_carbon

# Recommendation model only
python -m avle.train_recommendation

# Prediction model only
python -m avle.train_prediction

# Evaluate all models
python -m avle.evaluate
```

**Expected output:**
```
INFO:avle - Training U-Net...
INFO:avle - U-Net training complete (256 images)
INFO:avle - Training carbon model...
INFO:avle - Carbon model training complete
...
INFO:avle - Ablation results saved to avle/results/ablation.json
```

You'll see progress bars for each model. Models may take a few minutes each.

### Verify Model Training

Check that weights were created:

```powershell
# Windows
dir backend\avle\weights\

# macOS/Linux
ls backend/avle/weights/
```

You should see files like:
dir backend\avle\weights\
---

## Running the Full Application

### Option 1: Run Everything Manually (Recommended for Development)

**Terminal 1 - Backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1  # or source venv/bin/activate
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm start
```

Both should start without errors. The frontend will open at `http://localhost:3000` and the backend API at `http://127.0.0.1:8000`.

### Option 2: Use the PowerShell Startup Script (Windows)

From the project root:

```powershell
.\backend\start_backend.ps1
```

This starts the backend on port 8004. Then manually start the frontend:

```powershell
cd frontend
npm start
```

### Option 3: One-Click Startup (Advanced)

Create a file called `start_all.ps1` in the project root:

```powershell
# start_all.ps1
# Start backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\venv\Scripts\Activate.ps1; python -m uvicorn server:app --host 127.0.0.1 --port 8000"

# Wait for backend to start
Start-Sleep -Seconds 3

# Start frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm start"

Write-Host "✓ Backend running on http://127.0.0.1:8000"
Write-Host "✓ Frontend running on http://localhost:3000"
```

Then run:
```powershell
.\start_all.ps1
```

---

## API Endpoints Reference

### Health Check

```http
GET /api/
```

Response:
```json
{
  "service": "AVLE-C",
  "version": "1.0.0",
  "status": "ok"
}
```

### Run Analysis

```http
POST /api/analyze
Content-Type: application/json

{
  "bbox": [-75.5, 40.0, -75.0, 40.5],
  "date_t1_start": "2022-01-01",
  "date_t1_end": "2022-12-31",
  "date_t2_start": "2023-01-01",
  "date_t2_end": "2023-12-31",
  "size": 384,
  "allow_blockchain": true,
  "use_synthetic": false
}
```

**Parameters:**
- `bbox`: [west, south, east, north] - GeometryBBox in decimal degrees
- `date_t1_start/end`: First time period (yyyy-mm-dd)
- `date_t2_start/end`: Second time period (yyyy-mm-dd)
- `size`: Patch size in pixels (default 384)
- `allow_blockchain`: Save result to blockchain (optional)
- `use_synthetic`: Use synthetic data for testing (optional)

### Get All Jobs

```http
GET /api/jobs
```

Returns list of all past analyses.

### Get Single Job

```http
GET /api/jobs/{id}
```

### Get Ablation Table (Model Comparison)

```http
GET /api/ablation
```

### Get Model Metrics & Versions

```http
GET /api/weights/info
```

### Blockchain Status

```http
GET /api/blockchain
GET /api/blockchain/verify/{tx_hash}
```

### Full API Documentation

See [docs/RESEARCH_PAPER_GUIDE.md](docs/RESEARCH_PAPER_GUIDE.md) for detailed API documentation.

---

## Troubleshooting

### Backend Won't Start

**Error:** `ModuleNotFoundError: No module named 'avle'`

**Solution:**
```bash
# Make sure you're in the backend directory
cd backend

# Activate venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Run from the project root, not backend/
cd ..
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

**Error:** `MONGO_URL environment variable not set`

**Solution:**
Create `backend/.env`:
```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=avle
```

**Error:** `Connection refused` (MongoDB)

**Solution:**
MongoDB is not running. Start it:

**Windows:**
```powershell
# MongoDB runs as a service, verify it's running:
Get-Service MongoDB
# Start if stopped:
Start-Service MongoDB
```

**macOS:**
```bash
brew services start mongodb-community
```

**Linux:**
```bash
sudo systemctl start mongodb
```

### Frontend Won't Start

**Error:** `npm: command not found`

**Solution:** Node.js isn't installed or not in PATH. Reinstall from https://nodejs.org/

**Error:** `Port 3000 is already in use`

**Solution:**
```powershell
# Find and kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use a different port
npm start -- --port 3001
```

**Error:** `REACT_APP_BACKEND_URL` pointing to wrong backend

**Solution:** Update `frontend/.env`:
```bash
REACT_APP_BACKEND_URL=http://127.0.0.1:8000
```

Then restart frontend (`npm start`).

### Model Training Fails

**Error:** `FileNotFoundError: data/seg_cache/`

**Solution:** The directory is auto-created. If you get this during training:
```bash
mkdir -p backend/avle/data/seg_cache
```

Then retry training.

**Error:** `CUDA out of memory`

**Solution:** Training is using GPU but ran out of memory. Use CPU instead or reduce batch size:
```bash
# Edit backend/avle/train_segmentation_real.py
# Change batch_size from 4 to 2
python -m avle.train_segmentation_real  # batch_size=2 will be much slower but use less memory
```

**Error:** Download timeout during training

**Solution:** Internet connection issue. Retry:
```bash
# The data will be cached after first download
python -m avle.train_all
```

### Port Already in Use

**Windows:**
```powershell
# Find what's using the port
netstat -ano | findstr :8000
# Kill the process
taskkill /PID <PID> /F
```powershell
# Find what's using the port
netstat -ano | findstr :8000
# Kill the process
taskkill /PID <PID> /F
**Windows:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
cd backend
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` prefix in your PowerShell terminal.

**Deactivate when done:**
```powershell-m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

# Train models
python -m avle.train_all
```

### Frontend

```powershell
# Setup
cd frontend
npm install

# Run
npm start

# Build for production
npm run build
```

### Database

```powershell
# Check MongoDB connection
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print(client.admin.command('ping'))"
```

---

## Next Steps

1. **Explore the API**: Once both servers are running, visit `http://localhost:3000`
2. **Run an analysis**: Use the web interface to analyze a region
3. **Check results**: View analysis results in MongoDB or via `/api/jobs`
4. **Review models**: See [docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md) for training details
5. **Read the research**: Check [docs/RESEARCH_PAPER_GUIDE.md](docs/RESEARCH_PAPER_GUIDE.md) for the science behind AVLE-C

---

## Additional Resources

- **API Documentation**: [docs/RESEARCH_PAPER_GUIDE.md](docs/RESEARCH_PAPER_GUIDE.md)
- **Model Training Guide**: [docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md)
- **Project Evolution**: [docs/PROJECT_EVOLUTION.md](docs/PROJECT_EVOLUTION.md)
- **Data Guide**: [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md)
- **Interview Prep**: [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md)

---

## Support

If you encounter issues not listed in troubleshooting:

1. Check the backend logs in the terminal
2. Check browser console (F12) for frontend errors
3. Verify all prerequisites are installed (`python --version`, `node --version`)
4. Ensure MongoDB is running (`mongo --version`)
5. Double-check environment variables in `.env` files

---

**Last Updated**: April 2026  
**Project**: GREENPETAL (AVLE-C)
