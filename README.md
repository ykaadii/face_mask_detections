## About the Project
 
**MaskReady** is an end-to-end deep learning pipeline that detects whether individuals are wearing face masks in real-time using a webcam feed. It combines the power of **MobileNetV2** (pre-trained on ImageNet) with a custom classification head, fine-tuned on a curated face mask dataset.
 
The system uses **OpenCV's DNN-based face detector** to first localize faces within a video frame, and then passes each detected face region through the trained mask classifier. Results are displayed with color-coded bounding boxes and confidence percentages — green for mask-on, red for no mask.
 
This project is designed with practicality in mind: lightweight enough to run on a standard laptop, yet accurate enough for real-world deployment scenarios.
 
---
 
## Features
 
- **Real-time detection** via webcam with low-latency inference
- **Transfer learning** with MobileNetV2 for high accuracy on limited data
- **Two-stage pipeline** — face detection → mask classification
- **Confidence scores** displayed alongside bounding box labels
- **Color-coded output** — Green (Mask) / Red (No Mask)
- **Data augmentation** during training for improved generalization
- **Batch inference** for detecting multiple faces per frame simultaneously
- **Modular codebase** — training and inference scripts are fully decoupled
---
 
## Demo
 
```
[INFO] starting video stream...
 
Frame output:
┌──────────────────────────────────┐
│                                  │
│   ┌──────────┐  ┌────────────┐  │
│   │  Mask:   │  │  No Mask:  │  │
│   │  98.72%  │  │  94.31%   │  │
│   └──────────┘  └────────────┘  │
│   (Green Box)    (Red Box)       │
│                                  │
└──────────────────────────────────┘
 
Press 'q' to quit.
```
 
> *Live video feed with real-time bounding boxes and confidence labels overlaid on detected faces.*
 
---
 
## Project Structure
 
```
MaskReady/
│
├── dataset/
│   ├── with_mask/          # Training images — faces with masks
│   └── without_mask/       # Training images — faces without masks
│
├── face_detector/
│   ├── deploy.prototxt                       # Face detector network config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detector weights
│
├── train_mask_detector.py  # Script to train the mask classification model
├── detect_mask_video.py    # Script for real-time video mask detection
├── mask_detector.model     # Saved trained model (HDF5 format)
└── README.md
```
 
---
 
## Model Architecture
 
The model uses a **two-stage pipeline**:
 
### Stage 1 — Face Detection
OpenCV's `dnn` module loads a pre-trained **SSD (Single Shot Detector)** with a ResNet-10 backbone, trained to detect human faces at multiple scales. Detections below a confidence threshold of **0.5** are filtered out.
 
### Stage 2 — Mask Classification
 
| Layer | Details |
|---|---|
| **Base Model** | MobileNetV2 (pre-trained on ImageNet, top removed) |
| **Input Shape** | (224, 224, 3) |
| **AveragePooling2D** | Pool size (7, 7) |
| **Flatten** | — |
| **Dense** | 128 units, ReLU activation |
| **Dropout** | 0.5 rate |
| **Output Dense** | 2 units, Softmax activation |
 
The MobileNetV2 base layers are **frozen** during training — only the custom head is trained. This speeds up training significantly and prevents overfitting on the relatively small dataset.
 
**Training Configuration:**
 
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | `1e-4` |
| LR Decay | `lr / epochs` |
| Loss Function | Binary Cross-Entropy |
| Epochs | 20 |
| Batch Size | 32 |
 
**Data Augmentation applied during training:**
 
- Random rotation (±20°)
- Random zoom (15%)
- Width & height shifts (20%)
- Shear transformations (15%)
- Horizontal flips
---
 
## Dataset
 
The dataset is organized into two classes:
 
| Class | Description |
|---|---|
| `with_mask` | Face images of individuals wearing masks |
| `without_mask` | Face images of individuals without masks |
 
Images are resized to **224×224** pixels and preprocessed using MobileNetV2's built-in `preprocess_input` function before training. An 80/20 stratified train-test split is applied.
 
> You can extend the dataset by adding more images to the respective folders and retraining the model.
 
---
 
## Installation
 
### Prerequisites
 
- Python 3.7 or higher
- A webcam (for real-time detection)
- pip package manager
### Step 1 — Clone the Repository
 
```bash
git clone https://github.com/your-username/MaskReady.git
cd MaskReady
```
 
### Step 2 — Create a Virtual Environment (Recommended)
 
```bash
python -m venv venv
 
# On Windows
venv\Scripts\activate
 
# On macOS/Linux
source venv/bin/activate
```
 
### Step 3 — Install Dependencies
 
```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn imutils
```
 
Or install all at once using a requirements file:
 
```bash
pip install -r requirements.txt
```
 
<details>
<summary> requirements.txt (click to expand)</summary>
```
tensorflow>=2.4.0
keras>=2.4.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
imutils>=0.5.4
```
 
</details>
---
 
## Usage
 
### 1. Training the Model
 
If you want to train the model from scratch on your own dataset:
 
```bash
python train_mask_detector.py
```
 
Before running, update the `DIRECTORY` variable in `train_mask_detector.py` to point to your dataset location:
 
```python
# train_mask_detector.py — Line ~25
DIRECTORY = r"path/to/your/dataset"
```
 
The script will:
1. Load and preprocess all images from the dataset
2. Apply data augmentation
3. Fine-tune MobileNetV2 with the custom head
4. Evaluate on the test split and print a classification report
5. Save the trained model as `mask_detector.model`
**Expected output during training:**
 
```
[INFO] loading images...
[INFO] compiling model...
[INFO] training head...
Epoch 1/20
...
[INFO] evaluating network...
              precision    recall  f1-score   support
   with_mask       0.99      0.99      0.99       138
without_mask       0.99      0.99      0.99       138
[INFO] saving mask detector model...
```
 
---
 
### 2. Running Real-Time Detection
 
Once the model is trained (or using the pre-trained `mask_detector.model`), start the live detection:
 
```bash
python detect_mask_video.py
```
 
- The script accesses your default webcam (`src=0`)
- Faces are detected and classified in real time
- **Green bounding box** = Mask detected
- **Red bounding box** = No mask detected
- Press **`q`** to quit the video stream
> To use a different camera or a video file, change `VideoStream(src=0)` to `VideoStream(src=1)` for a secondary camera, or replace `VideoStream` with `cv2.VideoCapture("path/to/video.mp4")`.
 
---
 
## Results
 
The model achieves strong performance on the test set:
 
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| With Mask | ~99% | ~99% | ~99% |
| Without Mask | ~99% | ~99% | ~99% |
 
> *Actual results may vary based on your dataset size and composition.*
 
---
 
## Configuration
 
Key parameters you can tune in `train_mask_detector.py`:
 
```python
INIT_LR = 1e-4    # Initial learning rate
EPOCHS = 20       # Number of training epochs
BS = 32           # Batch size
```
 
Key parameter in `detect_mask_video.py`:
 
```python
confidence > 0.5  # Minimum confidence threshold for face detection
```
 
Adjust the confidence threshold to trade off between sensitivity (catching more faces) and precision (fewer false positives).
 
---
 