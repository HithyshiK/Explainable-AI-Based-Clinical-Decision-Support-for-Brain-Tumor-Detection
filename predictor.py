# predictor.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model 
import json
from PIL import Image
import cv2 

# =======================================================================
# 1. SETUP AND ARTIFACT LOADING
# =======================================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
IMG_SIZE = (224, 224)

# Load the class names
try:
    with open('class_names.json', 'r') as f:
        CLASS_NAMES = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Error: 'class_names.json' not found.")

# Load the advisory data (Using triple quotes to prevent SyntaxErrors)
tumor_guidance = {

    # ============================================================
    #                         GLIOMA
    # ============================================================
    "Glioma": {
        "description": "Gliomas are tumors that arise from glial cells of the brain or spinal cord. They can be low-grade (slow growing) or high-grade (aggressive).",
        "treatment": "Standard treatment includes maximal safe surgical removal, followed by radiation or chemotherapy depending on the grade.",
        "precautions": "Regular MRI follow-up, managing seizures (if present), good sleep, reduced stress, and early reporting of headaches or neurological symptoms."
    },

    # ============================================================
    #                         NORMAL
    # ============================================================
    "Normal": {
        "description": "No tumor or abnormal lesion detected. Brain structures appear healthy and symmetrical.",
        "treatment": "No treatment required.",
        "precautions": "Maintain a healthy lifestyle—proper sleep, hydration, stress control, and routine health checkups."
    },

    # ============================================================
    #                       MENINGIOMA
    # ============================================================
    "Meningioma": {
        "description": "A usually benign tumor that arises from the meninges (protective layers of the brain). Can occasionally be atypical or aggressive.",
        "treatment": "Surgery is the main treatment. Radiation therapy may be required for incomplete removal or atypical/anaplastic types.",
        "precautions": "Regular scans to monitor regrowth, avoid head trauma, and report symptoms like vision issues or balance problems early."
    },

    # ============================================================
    #                       SCHWANNOMA
    # ============================================================
    "Schwannoma": {
        "description": "A benign tumor that grows from Schwann cells of cranial or peripheral nerves. Commonly affects the vestibular (hearing/balance) nerve.",
        "treatment": "Microsurgery or stereotactic radiosurgery (e.g., Gamma Knife). Small tumors may only need monitoring.",
        "precautions": "Hearing tests, monitoring balance, avoiding loud noise, and regular MRI scans for growth tracking."
    },

    # ============================================================
    #                      NEUROCYTOMA
    # ============================================================
    "Neurocitoma": {
        "description": "A rare neuronal tumor typically found inside or near the brain ventricles. Usually low-grade but can obstruct CSF flow.",
        "treatment": "Surgical removal is preferred. Radiation may be used if tumor is partially removed or recurs.",
        "precautions": "Monitoring for symptoms of raised intracranial pressure (headache, nausea, vision changes) and routine MRI follow-up."
    }
}

# Load the H5 model file
try:
    MODEL = tf.keras.models.load_model('final_MBTC_model.h5', compile=True)
except Exception as e:
    raise RuntimeError(f"Error loading final_MBTC_model.h5: {e}")

# =======================================================================
# 2. PREDICTION FUNCTIONS
# =======================================================================

def preprocess_image(image_file):
    """Loads and preprocesses an uploaded image for model prediction."""
    # Rewind the file pointer to the beginning before reading.
    image_file.seek(0)
    
    img = load_img(image_file, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

def get_prediction(img_array):
    """Makes a prediction using the loaded model."""
    predictions = MODEL.predict(img_array)
    predicted_index = int(np.argmax(predictions))
    
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions)

    # Safely access the dictionary
    details = tumor_guidance.get(predicted_label, {
        "description": "No detailed advisory data available for this specific class.",
        "treatment": "Consult a medical professional.",
        "precautions": "Follow up with a specialist immediately."
    })
    
    return predicted_label, confidence, details

# =======================================================================
# 3. GRAD-CAM XAI FUNCTIONS
# =======================================================================

def find_last_conv_layer(model):
    """Automatically find the last convolutional layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if "conv" in layer.name.lower():
            return layer.name
    return 'top_conv'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates the class activation heatmap."""
    
    grad_model = Model(
        model.input,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if isinstance(preds, list):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        pred_index = int(pred_index)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_gradcam(uploaded_file, predicted_index):
    """
    Generates Grad-CAM visualization from the uploaded file.
    """
    # Load original for display/return
    uploaded_file.seek(0)
    original_img_pil = Image.open(uploaded_file)
    original_img_cv = np.array(original_img_pil.resize(IMG_SIZE)).astype(np.uint8)
    
    # Handle Color Channels for Display
    if len(original_img_cv.shape) == 2:
        original_img_cv = cv2.cvtColor(original_img_cv, cv2.COLOR_GRAY2RGB)
    elif original_img_cv.shape[2] == 4:
        original_img_cv = cv2.cvtColor(original_img_cv, cv2.COLOR_RGBA2RGB)

    try:
        # 1. Prepare input for model
        img_array = preprocess_image(uploaded_file)
        
        # 2. Set the layer name directly
        last_conv_layer_name = 'top_conv'
        
        # 3. Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, MODEL, last_conv_layer_name, pred_index=predicted_index)
        
        # 4. Superimposition
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        heatmap = np.uint8(255 * heatmap)
        
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2BGR) 
        
        # Blend
        superimposed_img = cv2.addWeighted(jet_heatmap, 0.6, img_bgr, 0.4, 0)
        
        # Convert back to RGB for Streamlit
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        return superimposed_img_rgb
        
    except Exception as e:
        print(f"GRAD-CAM ERROR: {e}")
        return original_img_cv