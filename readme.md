# F3 SEISMIC OVERWATCH

![Magnora Architecture](https://img.shields.io/badge/Architecture-LSTM_Neural_Network-00ffcc?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)
![UI](https://img.shields.io/badge/UI-Dash_Plotly-ff007f?style=for-the-badge)

An advanced, real-time deep learning dashboard designed to forecast and analyze streaming seismic telemetry. Built specifically to process and predict amplitude responses from the **F3 block in the Dutch sector of the North Sea**.

## 🌊 Geologic Context: The F3 Dataset
This project ingests 3D seismic data originally acquired to explore oil and gas in the Upper-Jurassic to Lower Cretaceous strata. The neural network specifically targets the upper 1200ms of the volume, mapping reflectors belonging to the **Miocene, Pliocene, and Pleistocene** epochs.

**Key features tracked by the telemetry stream:**
* Deposits from the massive Eridanos fluviodeltaic system.
* High-porosity (20-33%) deltaic sand and shale packages.
* Large-scale sigmoidal bedding (downlap, toplap, onlap structures).
* Bright spots indicating biogenic gas pockets.
* Facies variations ranging from transparent (uniform lithology) to chaotic (slumped deposits) and shingles (sandy turbidites).

## 🚀 Architecture Overview
The system bridges high-performance Long Short-Term Memory (LSTM) time-series forecasting with a hardware-accelerated, cyberpunk-aesthetic web interface. 

1. **Data Pipeline (`notebooks/01_eda.py`, `02_feature_engineering.py`)**: 
   Ingests raw `.sgy` files, cleans dead traces, clips anomalies to the 99.9th percentile, and scales variance. Engineers continuous time-series rolling windows and lag features.
2. **AI Core (`notebooks/03_training.py`)**: 
   A multi-layer LSTM neural network trained utilizing `ModelCheckpoint` and `EarlyStopping` to prevent overfitting.
3. **Telemetry Control Room (`app/app.py`)**: 
   A Dash/Plotly application featuring glassmorphism UI, real-time tracking, amplitude distributions, and T+1 forward predictions dynamically inferred by the `.keras` model.

## 🛠️ Installation & Setup

1. **Clone the repository and structure directories:**
```bash
├── app/
│   └── app.py
├── data/
│   └── f3_dataset.sgy (Download via Kaggle API)
├── models/
│   ├── dl_forecast_model.keras
│   └── data_scaler.pkl
├── outputs/
│   ├── processed_features.csv
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   └── 03_training.py
├── requirements.txt
└── README.md
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
pip install segyio
```

3. **Train the Model (Optional - Cloud GPU Recommended):**
Execute the Jupyter pipeline sequentially to process the data and compile the network.

4. **Download and Launch the Control Room:**
```bash
git clone https://github.com/Arya-azimi/F3-SEISMIC-OVERWATCH.git

cd F3-SEISMIC-OVERWATCH

cd app

python app/app.py
```
*Navigate to `http://127.0.0.1:8050` to interact with the live telemetry stream.*

## 📚 References
* Overeem, I., et al. (2001). *The Late Cenozoic Eridanos delta system in the Southern North Sea basin.* Basin Research, 13, 293-312.
* Sørensen, J.C., et al. (1997). *High frequency sequence stratigraphy of upper Cenozoic deposits.* Mar. Petrol. Geol., 14, 99-123.
* Dataset provided by Gustavo Scholze via [Kaggle](https://www.kaggle.com/datasets/gustavoscholze/f3-dataset) and OpendTect / TerraNubis.
