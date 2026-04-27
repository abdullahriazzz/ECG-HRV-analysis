import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.io as sio
import plotly.graph_objects as go
from fpdf import FPDF
import datetime
import tempfile
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="ECG & HRV Analysis Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f4f7fc !important; }
    .reportview-container { background: #f4f7fc; }
    h1, h2, h3, h4, p, span, div { color: #1e293b; font-family: 'Inter', sans-serif; }
    .metric-card {
        background: #ffffff; 
        border-radius: 16px; 
        padding: 20px; 
        border: none;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); }
    .metric-icon { font-size: 36px; }
    .metric-text { text-align: right; }
    .metric-value { font-size: 28px; font-weight: 800; color: #4f46e5; }
    .metric-label { font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 700; margin-top: 5px; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

st.title("ECG and HRV Analysis Dashboard")
st.markdown("Interactive clinical dashboard for time-domain, frequency-domain, and non-linear HRV analysis.")

# ==========================================
# 2. SIDEBAR INTERACTIVITY (FILTERS & ZOOM)
# ==========================================
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload custom ECG (.mat)", type=["mat"])
custom_fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=360, min_value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.header("👤 Patient Demographics")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_age = st.sidebar.number_input("Age", 1, 120, 45)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
patient_id = st.sidebar.text_input("Patient ID", "PID-10293")

st.sidebar.markdown("---")

st.sidebar.header("🎛️ Signal Processing Settings")
st.sidebar.subheader("Bandpass Filter")
lowcut = st.sidebar.slider("Lowcut Frequency (Hz)", 0.1, 5.0, 0.5, 0.1, help="Removes baseline wander")
highcut = st.sidebar.slider("Highcut Frequency (Hz)", 15.0, 100.0, 40.0, 1.0, help="Removes high-frequency noise")

st.sidebar.header("🔍 Visualization Settings")
zoom_range = st.sidebar.slider("ECG View Window (Seconds)", 0, 60, (0, 10), help="Select the time window to view the raw ECG")

# ==========================================
# 3. DATA GENERATION & SIGNAL PROCESSING
# ==========================================
@st.cache_data
def load_data():
    fs = 360
    duration = 300 # 5 minutes
    t = np.arange(0, duration, 1/fs)
    
    # Simulate RR intervals with realistic HRV
    mean_rr = 60 / 70  # ~70 bpm
    num_beats = int(duration / mean_rr) + 10
    
    # Random walk for RR intervals to create LF/HF spectral properties
    rr_intervals = np.random.normal(mean_rr, 0.02, num_beats)
    # Add slow respiratory sinus arrhythmia (RSA) ~ 0.25 Hz for HF band
    rsa = 0.05 * np.sin(2 * np.pi * 0.25 * np.cumsum(rr_intervals))
    # Add slower Mayer waves ~ 0.1 Hz for LF band
    mayer = 0.04 * np.sin(2 * np.pi * 0.1 * np.cumsum(rr_intervals))
    
    rr_intervals = rr_intervals + rsa + mayer
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < duration - 1]
    
    ecg = np.zeros_like(t)
    
    # Mathematical PQRST template
    def pqrst(t_local):
        p = 0.15 * np.exp(-((t_local + 0.15)**2) / (2 * 0.02**2))
        q = -0.1 * np.exp(-((t_local + 0.03)**2) / (2 * 0.015**2))
        r = 1.2 * np.exp(-((t_local)**2) / (2 * 0.015**2))
        s = -0.2 * np.exp(-((t_local - 0.04)**2) / (2 * 0.015**2))
        t_wave = 0.3 * np.exp(-((t_local - 0.2)**2) / (2 * 0.04**2))
        return p + q + r + s + t_wave
        
    for bt in beat_times:
        idx = int(bt * fs)
        half_len = int(0.4 * fs)
        start = max(0, idx - half_len)
        end = min(len(t), idx + half_len)
        t_local = t[start:end] - bt
        ecg[start:end] += pqrst(t_local)
        
    # Add realistic baseline wander and measurement noise
    wander = 0.3 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.sin(2 * np.pi * 0.02 * t)
    noise = np.random.normal(0, 0.02, len(t))
    ecg_raw = ecg + wander + noise
    
    return t, ecg_raw, fs

if uploaded_file is not None:
    try:
        mat_data = sio.loadmat(uploaded_file)
        valid_keys = [k for k in mat_data.keys() if not k.startswith('_')]
        
        best_arr = None
        max_size = 0
        
        for k in valid_keys:
            data = mat_data[k]
            
            # Handle structured MATLAB arrays (like the MathWorks ECGData format)
            if isinstance(data, np.ndarray) and data.dtype.names:
                if 'Data' in data.dtype.names:
                    data = data['Data']
                elif 'val' in data.dtype.names:
                    data = data['val']
            
            # Deep unwrap nested MATLAB structs/objects
            while isinstance(data, np.ndarray) and data.size == 1 and data.dtype == object:
                data = data.item()
                
            if isinstance(data, np.ndarray):
                arr = np.squeeze(data)
                if arr.size > max_size:
                    best_arr = arr
                    max_size = arr.size
                    
        if best_arr is not None and max_size > 100:
            ecg_raw = best_arr
            
            # If the data is multi-channel (e.g. 12-lead ECG), extract just the first lead
            if ecg_raw.ndim > 1:
                if ecg_raw.shape[0] > ecg_raw.shape[1]:
                    ecg_raw = ecg_raw[:, 0]
                else:
                    ecg_raw = ecg_raw[0, :]
                    
            ecg_raw = ecg_raw.astype(float)
            fs = custom_fs
            
            # Truncate to 10 seconds
            max_samples = int(10 * fs)
            ecg_raw = ecg_raw[:max_samples]
            
            t = np.arange(len(ecg_raw)) / fs
            st.sidebar.success(f"Successfully loaded {len(ecg_raw)} samples (Max 10s).")
        else:
            st.error("Uploaded MAT file contains no valid numerical data keys.")
            st.stop()
    except Exception as e:
        st.error(f"Critical error processing MAT file: {e}")
        st.stop()
else:
    t, ecg_raw, fs = load_data()

# Apply Butterworth Bandpass Filter
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
b, a = signal.butter(3, [low, high], btype='band')
ecg_filtered = signal.filtfilt(b, a, ecg_raw)

# Pan-Tompkins style preprocessing for ultra-robust peak detection
diff_ecg = np.diff(ecg_filtered, prepend=ecg_filtered[0])
squared_ecg = diff_ecg ** 2
window_len = max(1, int(0.15 * fs))
moving_avg = np.convolve(squared_ecg, np.ones(window_len)/window_len, mode='same')

# Find peaks on the moving average
# Use prominence (local height relative to surroundings) instead of absolute height
# This guarantees we catch every peak even if the absolute amplitude varies wildly
prominence_thresh = np.std(moving_avg) * 0.1
peaks_ma, _ = signal.find_peaks(moving_avg, distance=fs*0.2, prominence=prominence_thresh)

if len(peaks_ma) < 5:
    peaks_ma, _ = signal.find_peaks(moving_avg, distance=fs*0.2, height=np.mean(moving_avg))

if len(peaks_ma) < 5:
    st.error("⚠️ Insufficient R-peaks detected in the data. Please check your filter settings or upload a cleaner ECG signal.")
    st.stop()

# Refine peak locations back to the original filtered ECG
peaks = []
search_window = int(0.1 * fs) # 100 ms window
for p in peaks_ma:
    start = max(0, p - search_window)
    end = min(len(ecg_filtered), p + search_window)
    # Use absolute value to catch inverted R-peaks flawlessly
    local_peak = start + np.argmax(np.abs(ecg_filtered[start:end]))
    peaks.append(local_peak)

peaks = np.unique(peaks)
t_peaks = t[peaks]

# ==========================================
# 4. HRV CALCULATIONS
# ==========================================
# RR Intervals in milliseconds
rr_intervals = np.diff(t_peaks) * 1000  

# Time-Domain Metrics
mean_rr = np.mean(rr_intervals)
hr = 60000 / mean_rr
sdnn = np.std(rr_intervals)
rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
pnn50 = (nn50 / len(rr_intervals)) * 100

# Frequency-Domain Metrics (Welch's Method)
# Interpolate RR intervals to regular 4Hz grid for PSD calculation
interp_fs = 4.0
t_interp = np.arange(t_peaks[1], t_peaks[-1], 1/interp_fs)
rr_interp = np.interp(t_interp, t_peaks[1:], rr_intervals)

f, psd = signal.welch(rr_interp, fs=interp_fs, nperseg=min(256, max(16, len(rr_interp))))
lf_mask = (f >= 0.04) & (f <= 0.15)
hf_mask = (f >= 0.15) & (f <= 0.40)

lf_power = np.trapezoid(psd[lf_mask], f[lf_mask])
hf_power = np.trapezoid(psd[hf_mask], f[hf_mask])
lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

# Non-Linear Metrics (Poincaré)
rr_n = rr_intervals[:-1]
rr_n1 = rr_intervals[1:]

import scipy.stats as stats
counts, _ = np.histogram(rr_intervals, bins='auto')
prob = counts / sum(counts)
shannon_entropy = stats.entropy(prob) if len(prob) > 0 else 0

# ==========================================
# 5. DASHBOARD LAYOUT & PLOTLY VISUALIZATIONS
# ==========================================

# --- ROW 1: Filtered ECG Waveform ---
st.subheader("Live Filtered ECG Waveform")
# Slice data based on zoom slider
start_idx = int(zoom_range[0] * fs)
end_idx = int(zoom_range[1] * fs)

fig_ecg = go.Figure()
fig_ecg.add_trace(go.Scatter(x=t[start_idx:end_idx], y=ecg_filtered[start_idx:end_idx], 
                             mode='lines', name='Filtered ECG', line=dict(color='#6366f1', width=2)))

# Add R-peak markers within the zoomed window
visible_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
fig_ecg.add_trace(go.Scatter(x=t[visible_peaks], y=ecg_filtered[visible_peaks], 
                             mode='markers', name='R-Peaks', marker=dict(color='#ec4899', size=6)))

fig_ecg.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20),
                      xaxis_title="Time (s)", yaxis_title="Amplitude (mV)",
                      plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#1e293b'), hovermode="x unified")
fig_ecg.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
fig_ecg.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
st.plotly_chart(fig_ecg, use_container_width=True)


# --- ROW 2: Statistical Readouts & Frequency Domain ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### Time-Domain HRV Metrics")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div class="metric-card"><div class="metric-icon">🤍</div><div class="metric-text"><div class="metric-value">{hr:.0f} bpm</div><div class="metric-label">Heart Rate</div></div></div>
        <div class="metric-card"><div class="metric-icon">⏱️</div><div class="metric-text"><div class="metric-value">{mean_rr:.1f} ms</div><div class="metric-label">Mean RR</div></div></div>
        <div class="metric-card"><div class="metric-icon">📈</div><div class="metric-text"><div class="metric-value">{sdnn:.1f} ms</div><div class="metric-label">SDNN</div></div></div>
        <div class="metric-card"><div class="metric-icon">📉</div><div class="metric-text"><div class="metric-value">{rmssd:.1f} ms</div><div class="metric-label">RMSSD</div></div></div>
        <div class="metric-card"><div class="metric-icon">🌀</div><div class="metric-text"><div class="metric-value">{shannon_entropy:.2f}</div><div class="metric-label">Shannon Entropy</div></div></div>
        <div class="metric-card"><div class="metric-icon">⚖️</div><div class="metric-text"><div class="metric-value">{lf_hf_ratio:.2f}</div><div class="metric-label">LF/HF Ratio</div></div></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Poincaré Plot (Non-Linear)")
    fig_poincare = go.Figure()
    fig_poincare.add_trace(go.Scatter(x=rr_n, y=rr_n1, mode='markers', 
                                      marker=dict(color='#6366f1', size=5, opacity=0.6), name='RR Pairs'))
    fig_poincare.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                               xaxis_title="RR_n (ms)", yaxis_title="RR_n+1 (ms)",
                               plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#1e293b'))
    fig_poincare.update_xaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    fig_poincare.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    st.plotly_chart(fig_poincare, use_container_width=True)

with col3:
    st.markdown("### Power Spectral Density (PSD)")
    fig_psd = go.Figure()
    fig_psd.add_trace(go.Scatter(x=f, y=psd, mode='lines', line=dict(color='#94a3b8'), name='PSD'))
    fig_psd.add_trace(go.Scatter(x=f[lf_mask], y=psd[lf_mask], fill='tozeroy', mode='none', fillcolor='rgba(99, 102, 241, 0.35)', name='LF Band'))
    fig_psd.add_trace(go.Scatter(x=f[hf_mask], y=psd[hf_mask], fill='tozeroy', mode='none', fillcolor='rgba(236, 72, 153, 0.35)', name='HF Band'))
    
    fig_psd.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                          xaxis_title="Frequency (Hz)", yaxis_title="Power Density",
                          plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#1e293b'), xaxis=dict(range=[0, 0.6], gridcolor='#f1f5f9', zeroline=False), yaxis=dict(gridcolor='#f1f5f9', zeroline=False))
    st.plotly_chart(fig_psd, use_container_width=True)


# --- ROW 3: Temporal HRV Dynamics ---
st.subheader("Temporal HRV Dynamics (RR Tachogram)")
fig_tach = go.Figure()
fig_tach.add_trace(go.Scatter(x=t_peaks[1:]/60, y=rr_intervals, mode='lines+markers', 
                              line=dict(color='#6366f1', width=1.5), marker=dict(size=4, color='#ec4899'), name='RR Interval'))
fig_tach.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20),
                       xaxis_title="Time (Minutes)", yaxis_title="RR Interval (ms)",
                       plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#1e293b'), hovermode="x unified")
fig_tach.update_xaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
fig_tach.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
st.plotly_chart(fig_tach, use_container_width=True)

# ==========================================
# 6. PDF CLINICAL REPORT GENERATION
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("📄 Export Clinical Report")

def generate_pdf_report(name, age, gender, pid, hr_val, mean_rr_val, sdnn_val, rmssd_val, shannon_val, lfhf_val, chart_data=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pw = 210  # page width
    
    # ==========================================================
    # TOP BANNER (Dark Navy with white text)
    # ==========================================================
    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 0, pw, 32, 'F')
    pdf.set_fill_color(99, 102, 241)
    pdf.rect(0, 32, pw, 2, 'F')
    
    pdf.set_y(6)
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, txt="CLINICAL HRV ANALYSIS REPORT", ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(200, 210, 240)
    pdf.cell(0, 5, txt=f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=True, align='C')
    
    pdf.set_y(40)
    
    # ==========================================================
    # PATIENT DEMOGRAPHICS CARD
    # ==========================================================
    card_y = pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, card_y, 190, 30, 'DF')
    
    pdf.set_y(card_y + 2)
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(10)
    pdf.cell(0, 6, txt="PATIENT DEMOGRAPHICS", ln=True)
    
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(10)
    pdf.cell(90, 5, txt=f"Name:  {name}", ln=False)
    pdf.cell(90, 5, txt=f"Patient ID:  {pid}", ln=True)
    pdf.cell(10)
    pdf.cell(90, 5, txt=f"Age:  {age} yrs", ln=False)
    pdf.cell(90, 5, txt=f"Gender:  {gender}", ln=True)
    
    pdf.set_y(card_y + 33)
    
    # ==========================================================
    # TABLE HELPER
    # ==========================================================
    def draw_section_header(title):
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, txt=title, ln=True)
    
    def draw_table_header():
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.set_draw_color(30, 58, 138)
        pdf.cell(90, 8, txt="  PARAMETER", border=1, fill=True, align='L')
        pdf.cell(50, 8, txt="VALUE", border=1, fill=True, align='C')
        pdf.cell(50, 8, txt="UNIT", border=1, fill=True, align='C')
        pdf.ln()
    
    def draw_data_row(label, value, unit, alt=False):
        if alt:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.set_draw_color(226, 232, 240)
        pdf.cell(90, 8, txt=f"  {label}", border=1, fill=True, align='L')
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(50, 8, txt=value, border=1, fill=True, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(50, 8, txt=unit, border=1, fill=True, align='C')
        pdf.ln()
    
    # ==========================================================
    # TIME-DOMAIN TABLE
    # ==========================================================
    draw_section_header("TIME-DOMAIN METRICS")
    draw_table_header()
    draw_data_row("Heart Rate", f"{hr_val:.1f}", "bpm", alt=False)
    draw_data_row("Mean RR Interval", f"{mean_rr_val:.1f}", "ms", alt=True)
    draw_data_row("SDNN", f"{sdnn_val:.1f}", "ms", alt=False)
    draw_data_row("RMSSD", f"{rmssd_val:.1f}", "ms", alt=True)
    pdf.ln(5)
    
    # ==========================================================
    # FREQUENCY / NON-LINEAR TABLE
    # ==========================================================
    draw_section_header("FREQUENCY AND NON-LINEAR METRICS")
    draw_table_header()
    draw_data_row("Shannon Entropy", f"{shannon_val:.3f}", "bits", alt=False)
    draw_data_row("LF/HF Ratio", f"{lfhf_val:.3f}", "ratio", alt=True)
    pdf.ln(5)
    
    # ==========================================================
    # CLINICAL INTERPRETATION BOX
    # ==========================================================
    box_y = pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, box_y, 190, 24, 'DF')
    
    pdf.set_y(box_y + 2)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(10)
    pdf.cell(0, 5, txt="CLINICAL INTERPRETATION GUIDELINES", ln=True)
    
    pdf.set_font("Arial", '', 7)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(10)
    pdf.cell(0, 4, txt="- SDNN > 50 ms typically indicates normal autonomic function.", ln=True)
    pdf.cell(10)
    pdf.cell(0, 4, txt="- RMSSD reflects parasympathetic (vagal) tone; lower values may indicate stress.", ln=True)
    pdf.cell(10)
    pdf.cell(0, 4, txt="- LF/HF Ratio balances sympathetic/parasympathetic activity (Normal ~ 1.0 - 2.0).", ln=True)
    
    # ==========================================================
    # BOTTOM BANNER FOOTER
    # ==========================================================
    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 280, pw, 17, 'F')
    pdf.set_fill_color(99, 102, 241)
    pdf.rect(0, 279, pw, 1, 'F')
    pdf.set_y(283)
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(200, 210, 240)
    pdf.cell(0, 4, txt="This report was generated computationally and does not replace professional medical diagnosis.  |  Biomedical Engineering and Diagnostics Division", ln=True, align='C')
    
    # ==========================================================
    # PAGE 2: DIAGNOSTIC WAVEFORMS (Matplotlib)
    # ==========================================================
    if chart_data is not None:
        pdf.add_page()
        
        # Page 2 Banner
        pdf.set_fill_color(30, 58, 138)
        pdf.rect(0, 0, pw, 25, 'F')
        pdf.set_fill_color(99, 102, 241)
        pdf.rect(0, 25, pw, 2, 'F')
        pdf.set_y(7)
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, txt="DIAGNOSTIC WAVEFORMS AND SPECTRAL ANALYSIS", ln=True, align='C')
        pdf.set_y(35)
        
        tmp_files = []
        
        # --- ECG Chart ---
        fig1, ax1 = plt.subplots(figsize=(9, 2.5))
        ax1.plot(chart_data['t'], chart_data['ecg'], color='#6366f1', linewidth=0.8)
        ax1.plot(chart_data['t_peaks'], chart_data['ecg_peaks'], 'o', color='#ec4899', markersize=4)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Amplitude (mV)', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.set_facecolor('white')
        ax1.grid(True, color='#e2e8f0', linewidth=0.5)
        fig1.tight_layout()
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp1.close()
        fig1.savefig(tmp1.name, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        tmp_files.append(tmp1.name)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, txt="Filtered ECG Waveform with R-Peak Detection", ln=True)
        pdf.image(tmp1.name, x=10, w=190)
        pdf.ln(5)
        
        # --- PSD Chart ---
        fig2, ax2 = plt.subplots(figsize=(9, 2.5))
        ax2.plot(chart_data['f'], chart_data['psd'], color='#94a3b8', linewidth=0.8)
        ax2.fill_between(chart_data['f_lf'], chart_data['psd_lf'], alpha=0.35, color='#6366f1', label='LF Band')
        ax2.fill_between(chart_data['f_hf'], chart_data['psd_hf'], alpha=0.35, color='#ec4899', label='HF Band')
        ax2.set_xlabel('Frequency (Hz)', fontsize=8)
        ax2.set_ylabel('Power Density', fontsize=8)
        ax2.set_xlim(0, 0.6)
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7)
        ax2.set_facecolor('white')
        ax2.grid(True, color='#e2e8f0', linewidth=0.5)
        fig2.tight_layout()
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp2.close()
        fig2.savefig(tmp2.name, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        tmp_files.append(tmp2.name)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, txt="Power Spectral Density (LF/HF Bands)", ln=True)
        pdf.image(tmp2.name, x=10, w=190)
        pdf.ln(5)
        
        # --- RR Tachogram ---
        fig3, ax3 = plt.subplots(figsize=(9, 2.5))
        ax3.plot(chart_data['t_rr'], chart_data['rr'], color='#6366f1', linewidth=0.8, marker='o', markersize=3, markerfacecolor='#ec4899', markeredgecolor='#ec4899')
        ax3.set_xlabel('Time (Minutes)', fontsize=8)
        ax3.set_ylabel('RR Interval (ms)', fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.set_facecolor('white')
        ax3.grid(True, color='#e2e8f0', linewidth=0.5)
        fig3.tight_layout()
        tmp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp3.close()
        fig3.savefig(tmp3.name, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        tmp_files.append(tmp3.name)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, txt="RR Interval Tachogram", ln=True)
        pdf.image(tmp3.name, x=10, w=190)
        
        # Page 2 Footer
        pdf.set_fill_color(30, 58, 138)
        pdf.rect(0, 280, pw, 17, 'F')
        pdf.set_fill_color(99, 102, 241)
        pdf.rect(0, 279, pw, 1, 'F')
        pdf.set_y(283)
        pdf.set_font("Arial", 'I', 7)
        pdf.set_text_color(200, 210, 240)
        pdf.cell(0, 4, txt=f"Patient: {name}  |  ID: {pid}  |  Page 2 of 2", ln=True, align='C')
        
        for f_path in tmp_files:
            os.remove(f_path)
    
    # Safe cross-platform file saving
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    pdf.output(tmp_path)
    
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    os.remove(tmp_path)
    return pdf_bytes

# Generate the raw bytes
try:
    chart_data = {
        't': t, 'ecg': ecg_filtered, 't_peaks': t[peaks], 'ecg_peaks': ecg_filtered[peaks],
        'f': f, 'psd': psd, 'f_lf': f[lf_mask], 'psd_lf': psd[lf_mask], 'f_hf': f[hf_mask], 'psd_hf': psd[hf_mask],
        't_rr': t_peaks[1:]/60, 'rr': rr_intervals
    }
    pdf_data = generate_pdf_report(patient_name, patient_age, patient_gender, patient_id, hr, mean_rr, sdnn, rmssd, shannon_entropy, lf_hf_ratio, chart_data)

    st.sidebar.download_button(
        label="Download PDF Hospital Report",
        data=pdf_data,
        file_name=f"{patient_name.replace(' ', '_')}_HRV_Report.pdf",
        mime="application/pdf",
        help="Download a professionally formatted clinical PDF report of the current dashboard analysis."
    )
except Exception as e:
    st.sidebar.error(f"PDF generation error: {e}")
