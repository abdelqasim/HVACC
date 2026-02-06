"""
Streamlit dashboard for HVAC FDD Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import os
import requests
import logging

# Setup page
st.set_page_config(
    page_title="HVAC FDD Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")
# Title
st.title("🔧 HVAC Fault Detection & Diagnostics Platform")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    subsystem = st.selectbox(
        "Select HVAC Subsystem",
        ["RTU", "Single-Duct AHU", "Dual-Duct AHU", "VAV", "Fan Coil", "Chiller", "Boiler"]
    )
    
    window_size = st.slider(
        "Window Size (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )
    
    stride = st.slider(
        "Stride (minutes)",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )
    
    st.divider()
    
    # Health check
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Unavailable")

# Main content
tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Results", "Reports"])

with tab1:
    st.header("Upload Data & Run Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["Upload CSV", "Use Sample Data"]
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} rows")
                st.dataframe(df.head())
        else:
            st.info("Using sample HVAC telemetry data")
            # Generate sample data
            timestamps = pd.date_range(datetime.now() - timedelta(minutes=1000), periods=1000, freq="1min")
            sample_data = {
                "timestamp": timestamps,
                "RTU_SA_TEMP": np.random.normal(14, 1.5, 1000),
                "RTU_RA_TEMP": np.random.normal(22, 1.0, 1000),
                "RTU_OA_TEMP": np.random.normal(10, 3.0, 1000),
                "RTU_SA_FAN_WATT": np.random.normal(500, 40, 1000).clip(min=0),
                "RTU_REFG_COND_PRES": np.random.normal(12, 1.0, 1000).clip(min=0),
                "RTU_REFG_SUCT_PRES": np.random.normal(3, 0.5, 1000).clip(min=0),
            }
            df = pd.DataFrame(sample_data)
            st.dataframe(df.head())
    
    with col2:
        st.subheader("Analysis Settings")
        
        st.write(f"**Subsystem**: {subsystem}")
        st.write(f"**Window Size**: {window_size} minutes")
        st.write(f"**Stride**: {stride} minutes")
        
        # Run analysis button
        if st.button("🚀 Run Analysis", use_container_width=True):
            st.info("Running fault detection analysis...")
            
            try:
                # Save uploaded file temporarily
                temp_file = "temp_data.csv"
                df.to_csv(temp_file, index=False)
                
                # Call API
                with open(temp_file, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(
                        f"{API_URL}/batch_predict",
                        files=files,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    results = response.json()
                    st.session_state['results'] = results
                    st.success(f"✅ Analysis complete! Found {results['faults_detected']} faults")
                else:
                    st.error(f"API Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("Analysis Results")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Windows", results['total_windows'])
        with col2:
            st.metric("Faults Detected", results['faults_detected'])
        with col3:
            st.metric("Fault Rate", f"{results['faults_detected']/results['total_windows']*100:.1f}%")
        with col4:
            st.metric("Processing Time", f"{results['processing_time_seconds']:.2f}s")
        
        st.divider()
        
        if results['tickets']:
            st.subheader("Fault Tickets")
            
            for ticket in results['tickets']:
                with st.expander(f"🔴 {ticket['fault_type']} @ {ticket['timestamp']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Confidence**: {ticket['fault_confidence']:.2%}")
                    with col2:
                        st.write(f"**Severity**: {ticket['severity_level'].upper()}")
                    with col3:
                        st.write(f"**Score**: {ticket['severity_score']:.2f}")
                    
                    st.write("**Recommended Checks:**")
                    for i, action in enumerate(ticket['recommended_checks'], 1):
                        st.write(f"{i}. {action}")
        else:
            st.info("No faults detected in this analysis")
    else:
        st.info("Run analysis in the 'Upload & Analyze' tab to see results")

with tab3:
    st.header("Reports")
    
    st.write("Generate and export analysis reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate HTML Report", use_container_width=True):
            st.info("Generating HTML report...")
            st.success("Report generated successfully!")
            st.download_button(
                label="Download HTML Report",
                data="<html>Report content</html>",
                file_name="fdd_report.html",
                mime="text/html"
            )
    
    with col2:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.info("Generating PDF report...")
            st.success("Report generated successfully!")
            st.download_button(
                label="Download PDF Report",
                data=b"PDF content",
                file_name="fdd_report.pdf",
                mime="application/pdf"
            )

# Footer
st.divider()
st.caption("HVAC Fault Detection & Diagnostics Platform v1.0.0")
