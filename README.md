# ECG and HRV Analysis Dashboard

A professional clinical dashboard built with Streamlit for real-time ECG signal processing and Heart Rate Variability (HRV) analysis. This tool is designed for biomedical engineers and health professionals to generate diagnostic-grade reports from ECG data.

## 🚀 Features

- **Advanced Signal Processing**: 3rd-order Butterworth bandpass filtering and Pan-Tompkins style R-peak detection.
- **Comprehensive HRV Metrics**:
  - **Time-Domain**: HR, Mean RR, SDNN, RMSSD.
  - **Frequency-Domain**: LF, HF, and LF/HF ratio using Welch's PSD.
  - **Non-Linear**: Shannon Entropy and Poincaré plots.
- **Interactive Visualizations**: Dynamic ECG waveforms, Power Spectral Density (PSD) plots, and RR Tachograms using Plotly.
- **Professional Clinical Reports**: Generate and download multi-page PDF reports including patient demographics, metric tables, and diagnostic graphs.
- **MAT File Support**: Robust ingestion for clinical `.mat` datasets.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Science**: NumPy, Pandas, SciPy
- **Visualization**: Plotly, Matplotlib
- **Reporting**: FPDF

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ECG-HRV-Analysis-Dashboard.git
   cd ECG-HRV-Analysis-Dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run OEL1.py
   ```

## 📄 License

This project is for educational and clinical research purposes.

---
*Developed for Biomedical Engineering & Diagnostics.*
