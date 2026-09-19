import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# ============================================================
# Add Project Root to Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import AnimalClassifier


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Animal Species Detector",
    page_icon="🐾",
    layout="wide"
)


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet18_finetuned_best.pth"
)


# ============================================================
# Check Model Exists
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "❌ Model file not found.\n\n"
        f"Expected location:\n`{MODEL_PATH}`"
    )

    st.stop()


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_model():

    return AnimalClassifier(
        str(MODEL_PATH)
    )


classifier = load_model()


# ============================================================
# Header
# ============================================================

st.title("🐾 Animal Species Detector")

st.markdown(
    """
    Upload an animal image and the deep learning model
    will predict its species.

    **Model:** Fine-tuned ResNet18  
    **Classes:** 90 animal categories
    """
)

st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Architecture:** ResNet18")
    st.write("**Task:** Multi-class classification")
    st.write("**Number of species:** 90")
    st.write("**Test Accuracy:** 92.78%")
    st.write("**Input Size:** 224 × 224")

    st.divider()

    st.write("**Confidence Threshold:** 80%")

    st.caption(
        "Predictions below the confidence threshold "
        "are flagged as uncertain."
    )

    st.divider()

    st.write(
        f"**Device:** `{classifier.device}`"
    )


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an animal image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        # ====================================================
        # Uploaded Image
        # ====================================================

        with col1:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        # ====================================================
        # Model Prediction
        # ====================================================

        with st.spinner(
            "🔍 Analyzing image..."
        ):

            prediction = classifier.predict(
                image,
                threshold=0.80
            )

        confidence = prediction["confidence"]

        # ====================================================
        # Prediction Result
        # ====================================================

        with col2:

            st.subheader("🔮 Prediction")

            if prediction["accepted"]:

                species = prediction["class"].title()

                st.success(
                    f"🐾 {species}"
                )

                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%"
                )

                st.progress(
                    min(confidence, 1.0)
                )

                if confidence >= 0.95:

                    st.caption(
                        "🟢 High model confidence"
                    )

                elif confidence >= 0.80:

                    st.caption(
                        "🟡 Moderate model confidence"
                    )

            else:

                st.warning(
                    "⚠️ The model is not confident enough "
                    "to identify this image as one of the "
                    "supported animal species."
                )

                st.metric(
                    label="Model Confidence",
                    value=f"{confidence * 100:.2f}%"
                )

                st.progress(
                    min(confidence, 1.0)
                )

                st.caption(
                    "The prediction confidence is below "
                    "the 80% acceptance threshold."
                )

    except Exception as e:

        st.error(
            "❌ Unable to process this image."
        )

        st.exception(e)


# ============================================================
# Information Section
# ============================================================

else:

    st.info(
        "👆 Upload an image to start prediction."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Animal Species Detection • "
    "PyTorch + ResNet18 + Streamlit"
)