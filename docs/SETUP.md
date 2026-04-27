# AVLE-C — Local Setup

## Prerequisites
* Python 3.11
* Node.js 20 + Yarn 1.x
* MongoDB running on `localhost:27017`
* ~3 GB free disk for dependencies + model weights

## Install
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
yarn install
```

Environment files are kept minimal and already configured:

`backend/.env`
```
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
```

`frontend/.env`
```
REACT_APP_BACKEND_URL=<your backend URL>
```

## One-time: train all models from scratch
```bash
cd backend
python -m avle.train_all
```
This downloads ~16 Sentinel-2 + ESA WorldCover tiles (cached to
`avle/data/seg_cache/`), trains the U-Net, Random Forest, MLP, XGBoost
(BAU/Mitigation/quantile CI) + ARIMA baseline, and writes the full
ablation table to `avle/results/ablation.json`.

Expected runtime: ~4–6 min on a CPU-only machine.

## Run
Supervisor handles the backend (`:8001`) and frontend (`:3000`) already.
After code changes, hot-reload picks up automatically.

## Manual restart
```
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
```

## Endpoints
See `RESEARCH_PAPER_GUIDE.md` for the API surface.
