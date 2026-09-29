# Production Deployment Guide: AI Extreme Weather Intelligence

This guide outlines the recommended production architecture for deploying the three core components of the system: the Python AI Pipeline, the Node.js API, and the React Frontend Dashboard.

## Architecture Overview

Given the computationally heavy nature of the PyTorch/Diffusion models, the backend should be decoupled into an asynchronous batch-inference pipeline and a lightweight REST API.

```text
[ NEPS-G / NWP Data Sources ]
           │
           ▼ (Cron Job: Every 6 Hours)
[ AWS EC2 / GCP Compute (GPU Instance) ]  <-- Python AI Pipeline (PyTorch)
           │
           ▼ (JSON Contract Upload)
[ AWS S3 Bucket / PostgreSQL Database ]   <-- Data Storage
           │
           ▼ (HTTP GET)
[ Google Cloud Run / AWS Fargate ]        <-- Node.js API (server.js)
           │
           ▼ (REST API)
[ Vercel / Netlify Edge CDN ]             <-- React Dashboard (Vite)
```

---

## 1. Deploying the Python AI Pipeline (Heavy Compute)

Weather models don't need to run continuously; they run when new NWP data drops (typically 4 times a day: 00Z, 06Z, 12Z, 18Z).

**Recommended Platform:** AWS EC2 (g4dn.xlarge) or GCP Compute Engine (NVIDIA T4/L4 GPU).
**Strategy:** Batch Inference via Cron.

1. **Setup the Environment:**
   Provision a Linux VM with NVIDIA drivers and CUDA installed.
   ```bash
   git clone https://github.com/bipladipsaha/weather.git
   cd weather/ai
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install xarray dask metpy cdsapi dgl diffusers
   ```

2. **Automate the Pipeline:**
   Modify `e2e_runner.py` to write its output to a Cloud Storage bucket or a Database instead of printing to `stdout`.
   
3. **Schedule the Job:**
   Set up a cron job to run the pipeline automatically after the NWP data is published.
   ```bash
   # Run pipeline at 02:00, 08:00, 14:00, 20:00 UTC daily
   0 2,8,14,20 * * * /usr/bin/python3 /path/to/weather/ai/pipeline/e2e_runner.py > /var/log/ai_pipeline.log
   ```

---

## 2. Deploying the Node.js API (Middleware)

The Node API (`server.js`) is lightweight. It simply reads the generated JSON from the AI pipeline and serves it to the frontend.

**Recommended Platform:** Google Cloud Run, AWS App Runner, or Render.
**Strategy:** Containerized Microservice.

1. **Modify `server.js`:**
   Instead of using `spawn` to run the Python script on the fly (which would cause a timeout on serverless platforms and crash without a GPU), modify the `app.get('/api/events')` endpoint to read the latest JSON payload from your Cloud Storage or Database.

2. **Deploy via Render or Heroku (Easiest):**
   - Connect your GitHub repo to Render.com.
   - Set the Root Directory to `weather-dashboard`.
   - Set the Build Command to `npm install`.
   - Set the Start Command to `node server.js`.
   - Add environment variables (e.g., `PORT=3001`).

---

## 3. Deploying the React Frontend (User Interface)

The Vite React app is completely static once built. It should be hosted on a global CDN for lightning-fast loading.

**Recommended Platform:** Vercel or Netlify.
**Strategy:** Static Edge Hosting.

1. **Update API Endpoints:**
   In `App.jsx`, ensure all `fetch()` calls point to your deployed Node.js API URL (e.g., `https://api.your-weather-app.com`) instead of `http://localhost:3001`.

2. **Deploy to Vercel:**
   - Go to [Vercel.com](https://vercel.com) and import your GitHub repository.
   - Set the Framework Preset to **Vite**.
   - Set the Root Directory to `weather-dashboard`.
   - Click **Deploy**. Vercel will automatically run `npm run build` and distribute the `dist/` folder globally.

---

## 4. Environment Variables Checklist

Make sure you configure the following secrets in your production environments:

- **Python Pipeline:** `CDSAPI_URL`, `CDSAPI_KEY` (for downloading historical ERA5 data).
- **Node API:** `DATABASE_URL` or `AWS_ACCESS_KEY_ID` (to read the AI JSON outputs), `CORS_ORIGIN` (to allow requests only from your Vercel frontend).
- **React Frontend:** `VITE_API_BASE_URL` (pointing to your Node API).
