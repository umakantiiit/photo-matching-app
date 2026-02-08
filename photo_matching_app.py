import streamlit as st
from deepface import DeepFace
import json
import tempfile
import os

# ==================== CACHE MODELS (loads ONLY ONCE per session) ====================
@st.cache_resource
def load_models():
    with st.spinner("Downloading & loading models for the first time... (10-20 seconds)"):
        DeepFace.build_model("ArcFace")
        _ = DeepFace.extract_faces(
            img_path="https://picsum.photos/id/64/200/200",
            detector_backend="yunet",
            enforce_detection=False
        )
    return "✅ Models ready"

# ==================== APP UI ====================
st.set_page_config(page_title="PHOTO MATCHING APP", page_icon="🧑‍🦰", layout="centered")
st.title("🧑‍🦰 PHOTO MATCHING APP")

st.markdown("Upload one clear reference photo + one test photo and click the button.")

# Load models once
load_models()

# File uploaders
ref_file = st.file_uploader("Reference Image (clear face of the person)", type=["jpg", "jpeg", "png"])
gal_file = st.file_uploader("Test Image (can have multiple people)", type=["jpg", "jpeg", "png"])

if st.button("🔍 Check Presence & Similarity", type="primary", use_container_width=True):
    if ref_file is None or gal_file is None:
        st.error("Please upload both images!")
        st.stop()

    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ref:
        tmp_ref.write(ref_file.getvalue())
        ref_path = tmp_ref.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_gal:
        tmp_gal.write(gal_file.getvalue())
        gal_path = tmp_gal.name

    # Show preview
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_file, caption="Reference", use_column_width=True)
    with col2:
        st.image(gal_file, caption="Test Image", use_column_width=True)

    # Run DeepFace (very fast after first load)
    with st.spinner("Comparing faces..."):
        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=gal_path,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False,
            silent=True
        )

    distance = result["distance"]
    threshold = 0.35
    similarity = round(1 - distance, 4)
    is_match = distance < result["threshold"]

    output = {
        "reference_path": ref_file.name,
        "gallery_path": gal_file.name,
        "person_present": is_match,
        "similarity_score": similarity,
        "distance": round(distance, 4),
        "threshold_used": result["threshold"],
        "method": "DeepFace (ArcFace + mtcnn)"
    }

    # Beautiful result
    st.subheader("Result")
    if is_match:
        st.success(f"✅ PERSON IS PRESENT (Similarity: {similarity})")
    else:
        st.error(f"❌ PERSON IS NOT PRESENT (Similarity: {similarity})")

    st.json(output)

    # Cleanup temp files
    os.unlink(ref_path)
    os.unlink(gal_path)

st.caption("First run loads the model once. All future checks are instant.")
