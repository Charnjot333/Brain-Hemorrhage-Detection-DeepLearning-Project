import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import pydicom

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model("final.keras")

labels = [
    'any','epidural','intraparenchymal',
    'intraventricular','subarachnoid','subdural'
]

# =====================================================
# WINDOWING FUNCTION
# =====================================================
def window_image(img, center, width):
    min_val = center - width // 2
    max_val = center + width // 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-6)
    return img

# =====================================================
# DICOM LOADER (CORRECT PIPELINE)
# =====================================================
def load_dicom(file):
    dcm = pydicom.dcmread(file.name)
    img = dcm.pixel_array.astype(np.float32)

    # Apply medical windowing
    brain = window_image(img, 40, 80)
    subdural = window_image(img, 80, 200)
    bone = window_image(img, 600, 2800)

    img = np.stack([brain, subdural, bone], axis=-1)
    return img

# =====================================================
# PNG LOADER
# =====================================================
def load_png(file):
    img = cv2.imread(file.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

# =====================================================
# VALIDATION FUNCTION (ONLY FOR PNG)
# =====================================================
def is_ct_scan(img):
    img = img.astype(np.float32)

    variance = np.var(img)

    if len(img.shape) == 3:
        ch1 = img[:, :, 0]
        ch2 = img[:, :, 1]
        ch3 = img[:, :, 2]

        diff12 = np.mean(np.abs(ch1 - ch2))
        diff23 = np.mean(np.abs(ch2 - ch3))

        color_diff = (diff12 + diff23) / 2
    else:
        color_diff = 0

    # Relaxed thresholds
    if variance < 0.001:
        return False

    if color_diff > 0.15:
        return False

    return True

# =====================================================
# MAIN PREDICTION FUNCTION
# =====================================================
def predict(file):
    if file is None:
        return "❌ No file uploaded", {}

    try:
        # Load image
        if file.name.endswith(".dcm"):
            img = load_dicom(file)
        else:
            img = load_png(file)

            #  Apply validation ONLY for PNG
            if not is_ct_scan(img):
                return "⚠️ Invalid Input: Please upload a CT scan image", {}

        # Resize
        img = cv2.resize(img, (299, 299))
        img = np.expand_dims(img, axis=0)

        # Prediction
        preds = model.predict(img)[0]

        # Diagnosis
        if preds[0] > 0.5:
            diagnosis = "🔴 Hemorrhage Detected"
        else:
            diagnosis = "🟢 No Hemorrhage Detected"

        # Output formatting
        result = {
            "Any Hemorrhage": float(preds[0]),
            "Epidural": float(preds[1]),
            "Intraparenchymal": float(preds[2]),
            "Intraventricular": float(preds[3]),
            "Subarachnoid": float(preds[4]),
            "Subdural": float(preds[5]),
        }

        return diagnosis, result

    except Exception as e:
        return f"❌ Error: {str(e)}", {}

# =====================================================
# 🎨 UI DESIGN
# =====================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Brain Hemorrhage Detection System  

    Upload a **CT Scan Image (PNG or DICOM)**  
    ⚠️ Only medical CT images are supported.
    """)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CT Scan (.png / .dcm)")

        with gr.Column():
            diagnosis_output = gr.Textbox(label="Diagnosis")
            prediction_output = gr.Label(label="Prediction Confidence")

    submit_btn = gr.Button("🔍 Analyze")

    submit_btn.click(
        fn=predict,
        inputs=file_input,
        outputs=[diagnosis_output, prediction_output]
    )

# =====================================================
# LAUNCH
# =====================================================
demo.launch()