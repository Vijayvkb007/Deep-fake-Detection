# DeepFake Novelty Detection

This repository presents a deepfake video detection framework leveraging spatial-temporal inconsistencies and physiological signal analysis.

## 🧠 Methodology

Our approach combines two major detection pipelines:

### 1. **Eulerian Video Amplification (EVA) Preprocessing**

We employ **Eulerian Video Magnification** to amplify subtle facial motions such as pulse signals that are hard to fake. The steps include:

- **Face extraction** using MTCNN and cropping to 224×224.
- **EVA processing** to amplify subtle blood flow signals.
- **Feature extraction** using a pre-trained ResNet-50 on amplified facial videos.
- **Classification** using a custom MLP classifier (`classifier.py`) on extracted features.

### 2. **3D CNN for Spatio-Temporal Drift Detection**

A lightweight 3D CNN architecture processes RGB frames and computed optical flow magnitudes:

- Input shape: `(B, 4, T, H, W)` with RGB + Optical Flow channels.
- Network: Two 3D conv layers with ReLU, max pooling, dropout.
- Loss: Binary Cross Entropy + Temporal Consistency Loss.

This detects anomalies in lip sync and frame transitions introduced by synthetic generation errors.

### 3. **Motion Amplification & Visualization**

We also utilize:

- `MotionAmplification.ipynb` and `.py`: Compute dense optical flow (Farnebäck) to visualize motion anomalies.
- `visualization.py`: Visualizes t-SNE/UMAP projections for real vs fake embeddings.

## 📌 Highlights

- ✅ EVA amplifies physiological features like heartbeat for robust fake detection.
- ✅ 3D CNN captures drift in lip sync and temporal inconsistencies.
- ✅ Dense optical flow flags sudden motion spikes common in deepfakes.
- ✅ All modules modularized and scriptable for reproducible experimentation.
