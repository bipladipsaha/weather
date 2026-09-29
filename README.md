# 🌪️ AI Extreme Weather Intelligence Platform

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![Node.js](https://img.shields.io/badge/Node.js-Backend-339933?logo=node.js)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![Python](https://img.shields.io/badge/Python-ML_Pipeline-3776AB?logo=python)

An advanced meteorological intelligence platform that leverages AI to downscale coarse climate data, predict extreme weather events, and assess vulnerability risks across infrastructure, population, and agriculture.

## 🏗️ Repository Structure

This repository is organized into distinct microservices and pipelines:

```text
📦 weather
 ┣ 📂 weather-dashboard/    # React/Tailwind Frontend Dashboard
 ┣ 📂 weather-backend/      # Node.js/Express API & Data Ingestion
 ┣ 📂 final_model/          # Core Machine Learning Models (Residual RF)
 ┣ 📂 ai/                   # AI Data Exploration & Processing Scripts
 ┣ 📂 docs/                 # Documentation and Reference Materials
 ┗ 📜 README.md             # Project Documentation
```

### 1. Frontend (`weather-dashboard/`)
A state-of-the-art React application built with Tailwind CSS and Recharts. Features 3D glassmorphism UI components, interactive Leaflet maps, and real-time data visualization for meteorological intelligence.
- **Tech Stack**: React, Vite, Tailwind CSS, React-Leaflet, Recharts

### 2. Backend (`weather-backend/`)
A robust Node.js backend that serves AI predictions, handles large-scale meteorological datasets (GeoJSON, TIFF), and simulates downscaling processes.
- **Tech Stack**: Express.js, child_process (Python integration), CORS

### 3. AI & ML Pipeline (`final_model/` & `ai/`)
Python-based pipelines for processing ERA5 and CHIRPS meteorological data, training Residual Random Forest models, and executing spatial downscaling tasks to generate high-resolution risk maps.
- **Tech Stack**: Python, NumPy, Matplotlib, Scikit-learn, Xarray

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v16+)
- Python (3.8+)
- npm or yarn

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd weather-backend
   npm install
   npm start
   ```
   *The backend will run on `http://localhost:3000`*

2. **Start the Frontend Dashboard**
   ```bash
   cd weather-dashboard
   npm install
   npm run dev
   ```
   *The frontend will run on `http://localhost:5173`*

---

## 🌟 Key Features
- **AI Rainfall Downscaling**: Dynamically downscales coarse ERA5 global fields to 0.05° local resolution.
- **Risk Intelligence**: Real-time vulnerability metrics for population, infrastructure, and agriculture.
- **7-Day Trend Analysis**: Automated trend extraction using 3D visual components.
- **Interactive Spatial Mapping**: High-contrast meteorological maps with multiple data layers (Truth vs. AI Prediction).

## 🛡️ License
This project is licensed under the MIT License.
