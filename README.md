#  Brain Hemorrhage Detection System

A deep learning–based web application that detects **intracranial hemorrhage (ICH)** from CT scan images (DICOM or PNG format).  
The system uses a trained CNN model (EfficientNet-based) and provides real-time predictions with confidence scores.

---

## Live Demo
🔗 [https://huggingface.co/spaces/Charanjot/Brain-Hemorrhage-Detector]
### Prediction Result
![Prediction](images/result.png)
---

##  Features

- Supports **DICOM (.dcm)** and **PNG images**
- Detects multiple hemorrhage types:
  - Epidural
  - Intraparenchymal
  - Intraventricular
  - Subarachnoid
  - Subdural
- Shows **confidence scores for each class**
- Provides final diagnosis:
  - 🔴 Hemorrhage Detected
  - 🟢 No Hemorrhage Detected
- Built-in **input validation layer** (rejects non-CT images)
- User-friendly interface using **Gradio**

---

## Model Details

- **Architecture:** EfficientNet (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Loss Function:** Binary Crossentropy
- **Evaluation Metric:** AUC (Area Under Curve)
- **Input Size:** 299 × 299 × 3

---

## Preprocessing Pipeline

Medical images are processed using **windowing techniques**:

DICOM Image  
↓  
Windowing (3 channels)  
- Brain Window  
- Subdural Window  
- Bone Window  
↓  
Stack into 3-channel image  
↓  
Resize → Model Input  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Pydicom  
- Gradio  
- NumPy  

---

---

## ⚠️ Note
Only CT scan images are supported.  
Results on other images are not reliable.

---

## 👩‍💻 Author
Charanjot Kaur
