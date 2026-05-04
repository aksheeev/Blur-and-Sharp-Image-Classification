# Blur vs Sharp Image Classification

### Dual-Branch Multi-Scale Feature Extraction + XGBoost

---

## 📌 Overview

This project implements an image classification system to detect whether an image is **blurry** or **sharp** using a combination of:

* **Spatial domain features (Branch A)**
* **Frequency domain features (Branch B - FFT)**

The final model uses **XGBoost** for efficient and interpretable classification.

---

## 🚀 Key Features

* Multi-scale spatial feature extraction (3×3, 5×5, 7×7 grids)
* Frequency analysis using Fast Fourier Transform (FFT)
* Dual-branch feature fusion
* Ablation study comparing multiple configurations
* High accuracy (~99%) on test dataset
* Single image prediction demo

---

## 🧠 Methodology

### 🔹 Branch A: Spatial Features

* Image is divided into patches at multiple scales:

  * 3×3 (coarse)
  * 5×5 (medium)
  * 7×7 (fine)
* From each patch, 6 features are extracted:

  * Laplacian Mean
  * Laplacian Variance
  * Tenengrad
  * NGLV (Normalized Gray Level Variance)
  * LBP Mean
  * LBP Variance
* Features are aggregated using:

  * Mean, Std, Max, Min
* Final output: **72 features**

---

### 🔹 Branch B: Frequency Features (FFT)

* Apply Hann window to reduce edge artifacts
* Perform 2D FFT
* Shift frequency spectrum to center
* Extract log magnitude spectrum
* Crop central region (32×32)
* Final output: **1024 features**

---

### 🔹 Dual Branch Fusion

* Combine:

  * Spatial features (72)
  * Frequency features (1024)

👉 Final feature vector: **1096 dimensions**

---

## 🤖 Model

* Algorithm: **XGBoost Classifier**
* Why XGBoost?

  * Efficient for tabular data
  * Fast training
  * Handles high-dimensional features well
  * No GPU required

---

## 📂 Dataset Structure

```
dataset/
  train/
    blurry/
    sharp/
  test/
    blurry/
    sharp/
```

* Train: ~10,000 images (balanced)
* Test: ~1,000 images (balanced)

---

## ⚙️ Pipeline

1. Load dataset
2. Extract features (Branch A + Branch B)
3. Scale features (StandardScaler)
4. Train model (XGBoost)
5. Evaluate using:

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * AUC

---

## 📊 Results

| Model                     | Accuracy | F1 Score | AUC    |
| ------------------------- | -------- | -------- | ------ |
| Dual - Multi-scale (Ours) | ~99%     | ~0.99    | ~0.999 |

---

## 🔍 Ablation Study

Compared:

* Branch A (Fixed grid)
* Branch A (Multi-scale)
* Branch B (FFT only)
* Dual (Fixed)
* Dual (Multi-scale - Proposed)

👉 Dual multi-scale model performs best.

---

## 🧪 Single Image Prediction

You can test any image:

```python
predict_single_image("test.jpg")
```

Output:

* Prediction (Blurry / Sharp)
* Confidence score
* Visualization

---

## ⚠️ Limitations

* May struggle with motion blur (directional blur)
* FFT computation is relatively slow
* Uses handcrafted features instead of deep learning

---

## 🔮 Future Improvements

* Add motion blur dataset
* Use CNN for feature learning
* Optimize FFT computation
* Deploy as web application

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-image
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## 💡 Key Insight

Combining **spatial + frequency features** provides a stronger representation than using either alone.

---

## 📌 Author

Developed as part of a computer vision project focusing on efficient blur detection using hybrid feature extraction techniques.

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!

