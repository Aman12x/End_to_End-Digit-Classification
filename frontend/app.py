import streamlit as st
import numpy as np
import requests
import pandas as pd
import altair as alt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("🖌️ MNIST Digit Classifier")
st.write("Draw a digit (0–9) below and click **Predict**!")

# Layout: two buttons side by side
col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("Predict")
with col2:
    reset_btn = st.button("Reset")

# If reset is clicked, clear session state (forces canvas reset)
if reset_btn:
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    st.session_state.canvas_key += 1  # change key to refresh canvas

# Drawing Canvas (key changes when reset is clicked)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.get('canvas_key', 0)}",
)

# Prediction logic
if predict_btn:
    if canvas_result.image_data is not None:
        # Convert canvas to PIL
        img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))
        img = img.convert("L")  # grayscale
        img = img.resize((28, 28))

        # ✅ Proper MNIST preprocessing
        img_array = np.array(img).astype(np.float32)
        img_array = 255.0 - img_array  # invert
        img_array = img_array / 255.0  # normalize
        img_flat = img_array.flatten().tolist()

        # Call FastAPI backend
        backend_url = "http://127.0.0.1:8000"
        response = requests.post(f"{backend_url}/predict", json={"pixels": img_flat})
        result = response.json()

        # Show processed input
        st.image(
            img.resize((140, 140)), caption="Processed Input", use_container_width=False
        )

        # Show prediction result
        st.success(
            f"Predicted Digit: {result['prediction']} (Confidence: {result['confidence']:.2f})"
        )

        # Build dataframe for chart
        probs = [result["probabilities"][str(i)] for i in range(10)]
        df = pd.DataFrame({"digit": list(range(10)), "probability": probs})
        df["highlight"] = [
            "Predicted" if i == result["prediction"] else "Other" for i in range(10)
        ]

        # Altair bar chart with highlight
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("digit:O", title="Digit"),
                y=alt.Y("probability:Q", title="Probability"),
                color=alt.condition(
                    alt.datum.highlight == "Predicted",
                    alt.value("green"),
                    alt.value("lightblue"),
                ),
            )
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("Please draw a digit before predicting.")
