import streamlit as st
import requests
import io
import os
from PIL import Image, ImageDraw, ImageFont

# Configuration & Environment Variables
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DAGSHUB_URL = os.getenv("DAGSHUB_URL", "https://dagshub.com/Abhishek-B16/railway-defect-detection-mlops")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", os.getenv("MLFLOW_URL", "http://127.0.0.1:5000"))
API_DOCS_URL = os.getenv("API_DOCS_URL", f"{API_URL.rstrip('/')}/docs")

st.set_page_config(
    page_title="RailGuard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar: System Status & Model Info ---
st.sidebar.title("🚂 RailGuard")
st.sidebar.markdown("---")

st.sidebar.subheader("System Status")
health_status = "🔴 API Offline"
model_status = "🔴 Not Loaded"
model_info = None

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    if health_resp.status_code == 200:
        health_status = "🟢 API Online"
        model_status = "🟢 Loaded"
except requests.exceptions.RequestException:
    pass

st.sidebar.text(health_status)
st.sidebar.text(f"Model: {model_status}")
st.sidebar.markdown("---")

if "🟢 API Online" in health_status:
    try:
        info_resp = requests.get(f"{API_URL}/model-info", timeout=2)
        if info_resp.status_code == 200:
            model_info = info_resp.json()
            st.sidebar.subheader("Model Information")
            st.sidebar.text(f"Model: {model_info.get('model_name')}")
            st.sidebar.text(f"Version: {model_info.get('version')}")
            st.sidebar.text(f"Alias: {model_info.get('alias')}")
            st.sidebar.text(f"Task: {model_info.get('task')}")
            st.sidebar.markdown("---")
    except requests.exceptions.RequestException:
        st.sidebar.error("Could not fetch model info.")

# --- MLOps Navigation Showcase ---
st.sidebar.subheader("MLOps Navigation")
st.sidebar.markdown(f"[📖 API Interactive Docs]({API_DOCS_URL})")
st.sidebar.markdown(f"[📦 DagsHub Dataset & Remote]({DAGSHUB_URL})")
st.sidebar.markdown(f"[📊 MLflow Tracking UI]({MLFLOW_UI_URL})")

# --- Main Interface ---
st.title("Railway Track Defect Detection System")
st.markdown("Upload a railway track image to detect defects such as cracks, flaking, spalling, and missing bolts.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    # --- Prediction ---
    if st.button("Detect Defects", type="primary"):
        if "🔴" in health_status and "Offline" in health_status:
            st.error("Cannot perform detection. FastAPI server is offline.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    # Reset pointer
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        detections = data.get("detections", [])
                        
                        with col2:
                            st.subheader("Annotated Image")
                            
                            # Draw bounding boxes
                            draw = ImageDraw.Draw(image)
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except IOError:
                                font = ImageFont.load_default()
                            
                            colors = {
                                "crack": "red",
                                "spalling": "orange",
                                "flaking": "yellow",
                                "bolts": "cyan",
                                "joints": "blue",
                                "sheling": "magenta"
                            }
                            
                            for det in detections:
                                bbox = det["bbox"]
                                label = det["class_name"]
                                conf = det["confidence"]
                                color = colors.get(label, "red")
                                
                                # Draw Rectangle
                                draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=color, width=3)
                                
                                # Draw Text
                                text = f"{label} {conf*100:.1f}%"
                                text_bbox = draw.textbbox((bbox["x1"], max(0, bbox["y1"]-20)), text, font=font)
                                draw.rectangle(text_bbox, fill=color)
                                draw.text((bbox["x1"], max(0, bbox["y1"]-20)), text, fill="black", font=font)
                                
                            st.image(image, use_container_width=True)
                            
                        st.markdown("---")
                        st.subheader(f"Detected Defects: {len(detections)}")
                        
                        if len(detections) == 0:
                            st.success("No defects detected! The track appears clear.")
                        else:
                            for i, det in enumerate(detections):
                                st.info(f"**{det['class_name'].capitalize()}** 🎯 Confidence: {det['confidence']*100:.0f}%")
                    else:
                        st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The server took too long to respond.")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the API. Is it running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")