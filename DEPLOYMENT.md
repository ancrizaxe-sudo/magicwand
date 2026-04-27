# GREENPETAL Deployment Guide

## Overview
This is a full-stack application requiring:
- **Backend**: Python FastAPI server (port 8001)
- **Frontend**: React app (port 3000)
- **Database**: MongoDB

## Deployment Options

### Option 1: Render (Recommended - Free Tier)
**Best for: Quick deployment with free tier**

#### Backend Deployment
1. Create account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a **New Web Service** for backend:
   - Name: `greenpetal-backend`
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn avle.server:app --host 0.0.0.0 --port $PORT`
   - Plan: Free

4. Add Environment Variables:
   ```
   MONGO_URL=<your MongoDB connection string>
   DB_NAME=greenpetal
   CORS_ORIGINS=*
   ```

5. Create a **MongoDB** service on Render (Free tier available)

#### Frontend Deployment
1. Create a **New Web Service** on Render:
   - Name: `greenpetal-frontend`
   - Root Directory: `frontend`
   - Build Command: `yarn install && yarn build`
   - Start Command: `npx serve -s build -l $PORT`
   - Plan: Free

2. Add Environment Variable:
   ```
   REACT_APP_BACKEND_URL=https://greenpetal-backend.onrender.com
   ```

---

### Option 2: Railway (Good Alternative)
**Best for: Simpler setup, generous free tier**

1. Create account at [railway.app](https://railway.app)
2. Deploy from GitHub
3. Add MongoDB plugin
4. Set environment variables same as above

---

### Option 3: Fly.io (Docker-based)
**Best for: More control, global distribution**

1. Install Fly CLI: `winget install flyctl`
2. Login: `fly auth login`
3. Create `Dockerfile` in root:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm

# Copy backend
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy frontend
COPY frontend/package.json frontend/yarn.lock ./
RUN npm install -g yarn && yarn install

COPY . .

EXPOSE 8001 3000

# Start both services
CMD python -m avle.server & cd frontend && yarn start
```

4. Deploy: `fly launch`

---

### Option 4: Vercel + Railway (Frontend + Backend Separate)

#### Frontend on Vercel (Recommended for React)
1. Install Vercel CLI: `npm i -g vercel`
2. In frontend folder: `vercel --prod`
3. Set environment variable `REACT_APP_BACKEND_URL` to your backend URL

#### Backend on Railway/Render
- Same as Option 1

---

## Required Changes for Deployment

### 1. Update Backend for Production
In `backend/avle/server.py`, ensure CORS allows your frontend domain:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Update Frontend API URL
In `frontend/src/lib/api.js`, update base URL:

```javascript
const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
```

### 3. Create Production Build
```bash
cd frontend
yarn build
```

---

## Quick Deploy Steps (Render)

1. **Push code to GitHub**
2. **Deploy Backend**:
   - Go to Render → New → Web Service
   - Connect GitHub repo
   - Configure: Root = `backend`, Build = `pip install -r requirements.txt`, Start = `uvicorn avle.server:app --host 0.0.0.0 --port $PORT`
3. **Deploy MongoDB**: Render → New → MongoDB
4. **Deploy Frontend**:
   - Render → New → Web Service
   - Configure: Root = `frontend`, Build = `yarn install && yarn build`, Start = `npx serve -s build -l $PORT`
5. **Set REACT_APP_BACKEND_URL** to your backend URL

---

## Getting Your Live Link

After deployment, you'll get URLs like:
- Backend: `https://greenpetal-backend.onrender.com`
- Frontend: `https://greenpetal-frontend.onrender.com`

Share the **Frontend URL** as your live link!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CORS errors | Update CORS_ORIGINS in backend |
| MongoDB connection | Use MongoDB Atlas or Render MongoDB |
| Build fails | Check Node.js version (20+) and Python (3.11) |
| Static assets not loading | Ensure `yarn build` runs before serve |

---

## Alternative: Use ngrok for Temporary Testing

For quick testing without full deployment:
```bash
# Backend
cd backend
uvicorn avle.server:app --port 8001

# In another terminal
ngrok http 8001
```

This gives you a temporary public URL.