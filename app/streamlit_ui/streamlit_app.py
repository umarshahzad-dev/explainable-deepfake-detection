r"""Run from the project root with: python -m streamlit run app/streamlit_ui/streamlit_app.py"""
import base64
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
import streamlit as st
from app.streamlit_ui.backend_client import analyze_upload, analyze_url, recent_history

st.set_page_config(page_title="Deepfake experiment", layout="wide")
st.title("Explainable deepfake facial detection")
st.caption("Still-image model scores. Heatmaps show model attribution, not proof of forgery.")
base = st.sidebar.text_input("Local API", "http://127.0.0.1:8000")
uploaded = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"])
url = st.text_input("Or a public HTTPS image URL")
explain = st.checkbox("Generate Grad-CAM", value=True)
if st.button("Analyze", type="primary"):
    if bool(uploaded) == bool(url.strip()):
        st.error("Choose one uploaded image or one URL.")
    else:
        try:
            with st.spinner("Analyzing faces"):
                if uploaded:
                    result = analyze_upload(base, uploaded.getvalue(), uploaded.type or "image/png", explain=explain)
                else:
                    result = analyze_url(base, url.strip(), explain=explain)
            st.session_state["result"] = result
        except Exception as error:
            st.error(str(error))

result = st.session_state.get("result")
if result:
    def display(data_url):
        st.image(base64.b64decode(data_url.split(",", 1)[1]), use_container_width=True)
    st.write(f"Model: {result['run_id']} | Calibration: {result['calibration_applied']}")
    if result["training_pilot"]:
        st.warning("Diagnostic pilot weights; these are not a validated detector.")
    display(result["image"]["preview_data_url"])
    if not result["faces"]:
        st.info("No faces detected. No REAL/FAKE verdict was produced.")
    for face in result["faces"]:
        with st.expander(f"Face {face['face_index'] + 1}: {face['label'] or face['status']}", expanded=True):
            st.write("Box:", face["box"])
            if face["status"] != "accepted":
                st.warning("Face did not meet the evaluated quality/size gate.")
                continue
            st.metric("FAKE model score", f"{face['fake_score']:.1%}")
            st.progress(face["fake_score"])
            if face.get("uncertain"):
                st.warning("Score is close to the selected decision threshold.")
            if face.get("overlay_data_url"):
                display(face["overlay_data_url"])
                if not face.get("explanation_informative"):
                    st.caption("No positive class-attribution region was found.")
    st.caption(f"Analysis took {result['elapsed_seconds']:.2f}s; threshold {result['threshold']:.3f}.")
if st.sidebar.button("Refresh history"):
    try:
        st.sidebar.dataframe(recent_history(base), use_container_width=True)
    except Exception as error:
        st.sidebar.error(str(error))
