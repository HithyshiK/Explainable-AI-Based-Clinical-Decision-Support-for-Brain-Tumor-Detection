# app.py

import streamlit as st
from PIL import Image
import predictor # Import the script containing your model and prediction functions

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Detection UI",
    page_icon="🧠",
    layout="centered"
)

# --- Title and Description ---
st.title("🧠 Brain Tumor Classification UI")
st.markdown("Upload a brain MRI scan image (T1, T1C+, or T2 sequence) for immediate classification.")

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose an MRI Image (jpg, jpeg, png)...",
    type=["jpg", "jpeg", "png"]
)

# app.py (Modified Section)

# ... (Start of the file is the same) ...

if uploaded_file is not None:
    # --- Display Uploaded Image ---
    st.subheader("Uploaded Scan")
    image = Image.open(uploaded_file)
    #st.image(image, caption='MRI Scan', use_column_width=True)
    st.image(image, caption='MRI Scan', use_container_width=True)
    
    # --- Prediction Button ---
    if st.button('Classify Scan', type="primary"):
        with st.spinner('Analyzing image and making prediction...'):
            try:
                # 1. Preprocess the image
                img_array = predictor.preprocess_image(uploaded_file)

                # 2. Get Raw Prediction (Needed for Grad-CAM index)
                predictions = predictor.MODEL.predict(img_array)
                predicted_index = np.argmax(predictions)
                
                # 3. Get Classification Details
                label, confidence, details = predictor.get_prediction(img_array)
                
                # 4. Generate Grad-CAM image
                gradcam_image = predictor.generate_gradcam(uploaded_file, predicted_index)


                # --- Display Results ---
                st.success("Analysis Complete!")
                
                # Create two columns for results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Classification")
                    confidence_percent = f"{confidence * 100:.2f}%"
                    
                    if "NORMAL" in label:
                        st.balloons()
                        st.metric(label="Predicted Status", value=label, delta="Healthy Scan")
                    else:
                        st.error(f"Predicted Tumor: **{label}**")
                        
                    st.metric(label="Confidence", value=confidence_percent)
                    
                    st.subheader("📋 Advisory Module")
                    st.markdown(f"**Description:** {details['description']}")
                    st.info(f"**💊 Recommended Treatment:** {details['treatment']}")
                    st.warning(f"**⚠️ Precautions:** {details['precautions']}")
                    st.caption("Disclaimer: This is an AI-generated analysis.")

                with col2:
                    st.subheader("🔥 Explainable AI (Grad-CAM)")
                    #st.image(gradcam_image, caption=f"Activation Heatmap for: {label}", use_column_width=True)
                    st.image(gradcam_image, caption=f"Activation Heatmap for: {label}", use_container_width=True)
                    st.markdown("The **heatmap** shows the regions (red/yellow) the model focused on to make this classification, increasing clinical trust.")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.markdown("Please ensure the uploaded image file is valid and all required artifacts (`.h5`, `.json`) are in the correct directory.")

# ... (End of the file is the same) ...